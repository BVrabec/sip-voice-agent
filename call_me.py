#!/usr/bin/env python3
"""call_me.py "<opening sentence>" — outbound SIP call with a spoken conversation.

Pipeline per turn: rx audio -> VAD -> 16 kHz mono wav -> OpenRouter STT ->
Hermes chat completions -> OpenRouter TTS -> 48 kHz stereo wav -> live call.

Speaking uses baresip's mixausrc module rather than baresipy's send_audio:
baresip 4.x stalls the runtime /ausrc switch that send_audio relies on (a
transmit-dump check showed only ~0.3 s of a 2 s clip reaching the wire), while
mixausrc_enc_start mixes a file into the live encoding stream and stops cleanly
for barge-in.

Reads configuration from the environment (call.sh sources ./config.env).
Never logs or prints credentials.
"""
import argparse
import base64
import json
import logging
import os
import pathlib
import re
import struct
import subprocess
import sys
import threading
import time

import requests
from baresipy import BareSIP
from baresipy.audio import resample_pcm16
from baresipy.utils.log import LOG

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import approval_voice as av          # noqa: E402  (same directory)

HERE = pathlib.Path(__file__).resolve().parent
CLIP_DIR = HERE / 'clips'
LOG_PATH = HERE / 'call_me.log'


# --- credential hygiene: baresipy echoes raw baresip lines, including the account line ---
class _MaskAuthPass(logging.Filter):
    _re = re.compile(r"(auth_pass=)[^;>'\s]*", re.I)

    def filter(self, record):
        try:
            msg = record.getMessage()
            masked = self._re.sub(r"\1***", msg)
            if masked != msg:
                record.msg = masked
                record.args = ()
        except Exception:
            pass
        return True


if LOG.handler is not None:
    LOG.handler.addFilter(_MaskAuthPass())


# --- serialise every write to baresip's pty -----------------------------------
# baresipy polls the call state (/listcalls, /callstat) from its own reader thread
# while we send speech commands. Two writers interleaving on the same pty corrupt
# each other: the first live run delivered a stray "p" and dumped baresip's command
# list mid-call. Every command now goes through one lock, baresipy's included.
_PTY_LOCK = threading.RLock()
_orig_do_command = BareSIP.do_command


def _locked_do_command(self, action):
    with _PTY_LOCK:
        return _orig_do_command(self, action)


BareSIP.do_command = _locked_do_command


class CallSIP(BareSIP):
    """BareSIP that does not hang up on its own.

    baresipy treats baresip's "failed to set audio-source (No such device)" as a
    voicemail and calls hang() (handle_error -> handle_audio_stream_failure). Our
    speech goes through mixausrc, and such a line must never end the call: log it
    and carry on instead.
    """

    def handle_audio_stream_failure(self) -> None:
        log('baresip reported an audio stream failure (not hanging up)')

    def handle_audio_source_error(self, error: str) -> None:
        log(f'baresip audio-source error (not hanging up): {error}')

    def handle_dtmf_received(self, char: str, duration: int = 0) -> None:
        """A key pressed on the phone's keypad, in any of the three transports.

        baresipy parses RFC 4733 telephone-events, SIP INFO and in-band tones and calls this
        for each one; the key is recorded here and the main loop decides what to do with it
        (with KEYPAD_INTERRUPT=1 it is the only thing that stops a reply).
        """
        note_key(char, duration)
        try:
            super().handle_dtmf_received(char, duration)
        except Exception:
            pass

    def _arm_idle_source(self) -> None:
        """Leave the call's audio source alone.

        baresipy points it at its own silence.wav -- 60 s, 8 kHz mono -- which both
        changes the stream format mid-call (baresip then fails the opus encoder with
        "opus encode error: 23040 samples", dropping the call) and hits EOF at 60 s.
        Our source is the 900 s 48 kHz stereo file set in ~/.baresipy/config, so the
        rearm is neither needed nor wanted.
        """

    def _schedule_idle_rearm(self) -> None:
        """No idle re-arm: see _arm_idle_source above."""


def envc(name: str, default: str = '') -> str:
    return os.environ.get(name, default).strip()


# --- settings ---
SIP_USER = envc('SIP_USER')
SIP_PASS = envc('SIP_PASS')
SIP_GATEWAY = envc('SIP_GATEWAY', 'sip.linphone.org')
SIP_TRANSPORT = envc('SIP_TRANSPORT', 'udp')
SIP_MEDIAENC = envc('SIP_MEDIAENC', 'srtp-mand')
TARGET_SIP = envc('TARGET_SIP')
# Spoken approvals. With PHONE_APPROVAL_ID set this call exists only to ask a question:
# it speaks it, judges the answer, writes it down and hangs up -- no agent, no memory,
# no transcript. Without it a normal call may still be asked to ask, mid-conversation.
APPROVAL_ID = envc('PHONE_APPROVAL_ID')
APPROVAL_ONLY = bool(APPROVAL_ID)
APPROVAL_TEXT = envc('PHONE_APPROVAL_TEXT')

# --- scheduled calls (the morning / evening / weekend jobs) -------------------
# PHONE_SCRIPT_FILE points at a small JSON spec: a prepared script to speak, a question
# to ask after it, an optional reminder to give before hanging up, and where to write
# the outcome (answered / answered_silent / not_answered) for the job to read.
SCRIPT_FILE = envc('PHONE_SCRIPT_FILE')
CLOSE_LINE = envc('PHONE_CLOSE_LINE', 'Goodbye.')
# The session key is how the approval gate knows this client can answer: without it a call
# looks like any other API client, which is refused instantly. It also scopes memory.
VOICE_SESSION_KEY = envc('PHONE_SESSION_KEY', 'phone-voice:main')
APPROVAL_INCALL_WAIT_S = float(envc('PHONE_APPROVAL_INCALL_WAIT_S', '30'))
OPENROUTER_KEY = envc('OPENROUTER_API_KEY')
HERMES_KEY = envc('HERMES_API_KEY')
HERMES_URL = envc('HERMES_API_URL', 'http://127.0.0.1:8642/v1/chat/completions')
HERMES_MODEL = envc('HERMES_MODEL', 'hermes-agent')
# Phone-call only overrides, sent per request. Empty means 'leave it as it is'.
PHONE_MODEL = envc('PHONE_MODEL')          # e.g. qwen/qwen3-235b-a22b-2507
PHONE_REASONING = envc('PHONE_REASONING')  # e.g. none / minimal / low
HOLD_LINE_S = float(envc('HOLD_LINE_S', '2.5'))   # say 'One moment.' after this long silent
HOLD_LINE_TEXT = envc('HOLD_LINE_TEXT', 'One moment.')
HOLD_SUPPRESS_S = float(envc('HOLD_SUPPRESS_S', '0.2'))   # he must be speaking this long for
                                                          # the holding line to stand down
HERMES_TIMEOUT = float(envc('HERMES_TIMEOUT', '120'))
STT_MODEL = envc('STT_MODEL', 'openai/whisper-large-v3-turbo')
# Resend the same audio if the first transcription has not answered in this long (one retry,
# whichever answer arrives first wins). 0 disables it.
STT_RETRY_S = float(envc('STT_RETRY_S', '3'))
STT_LANG = envc('STT_LANG', 'en')
TTS_MODEL = envc('TTS_MODEL', 'elevenlabs/eleven-flash-v2.5')
TTS_VOICE = envc('TTS_VOICE', 'sarah')
# Speech pace. >1 speaks more slowly (1.2 = 20% slower); 1.0 is the normal pace. The slowdown
# is a local ffmpeg time-stretch -- pitch kept, no extra provider cost, no extra request.
# The scheduled call sets it per call from its script spec ('tts_slow'); the env var is the
# fallback for a call started by hand. Used first by the weekday morning briefing (17 Oct).
TTS_SLOW = float(envc('TTS_SLOW') or '1.0')
VAD_RMS = int(envc('VAD_RMS', '800'))
# Barge-in must be a real voice, not room noise or our own echo. Measured on the 16:50 call:
# the echo of our own playback peaked at rms 1912, the owner's speech sat at 3900-5344.
# A gate of 1600 therefore sat below the echo and told us nothing.
BARGIN_RMS = int(envc('BARGIN_RMS', '2500'))
# How many seconds of the owner's audio above the gate count as an interruption. Measured as
# seconds of HIS audio, not elapsed time and not loop iterations: the read loop's cadence is not
# fixed, so a 0.3 s interruption can arrive as one chunk. 0.15 s is safe on headphones, where
# there is no echo of our own playback to trip the gate.
BARGE_MIN_S = float(envc('BARGE_MIN_S', '0.3'))
BARGIN_WINDOW_S = float(envc('BARGIN_WINDOW_S', '1.2'))
# F: while we are speaking, only a key pressed on the phone's keypad may stop the reply -- the
# volume trigger is off, so nothing said on the line (or near it) interrupts. Off by default;
# set KEYPAD_INTERRUPT=1 in config.env to enable it. baresipy hands key events to
# CallSIP.handle_dtmf_received() for all three transports (RFC 4733 telephone-event, SIP INFO
# and in-band tones), which is what makes this possible at all.
KEYPAD_INTERRUPT = envc('KEYPAD_INTERRUPT', '0').lower() not in ('0', 'false', 'no', 'off')
KEY_STOP_MAX_S = float(envc('KEY_STOP_MAX_S', '0.2'))        # a key must stop speech this fast
SPEAK_END_GRACE_S = float(envc('SPEAK_END_GRACE_S', '0.3'))  # listen this long after we finish
_KEY_LOCK = threading.Lock()
_key_events = []                    # [(when, char)] newest last


def note_key(char: str, duration: int = 0) -> None:
    """Record a key pressed on the phone's keypad (called from baresip's reader thread)."""
    with _KEY_LOCK:
        _key_events.append((time.time(), char))
        del _key_events[:-8]


def clear_keys() -> None:
    with _KEY_LOCK:
        _key_events.clear()


def keypad_stop(playing_since: float, now: float, *, speaking: bool,
                approval_active: bool) -> tuple:
    """(stop?, milliseconds, keys) for keys pressed while the current clip was playing.

    Nothing to do with the audio: a key press makes no sound on the line, so this is checked
    before the level. During an approval question keys are not an answer and are ignored.
    """
    if not speaking or approval_active:
        return (False, None, '')
    with _KEY_LOCK:
        fresh = [e for e in _key_events if e[0] >= (playing_since or 0)]
    if not fresh:
        return (False, None, '')
    return (True, int(round((now - fresh[0][0]) * 1000)), ''.join(c for _, c in fresh))


def speak_grace_active(speak_ended_at: float, now: float) -> bool:
    """True while the short pause after our own speech lasts: not a turn yet."""
    return bool(speak_ended_at) and (now - speak_ended_at) < SPEAK_END_GRACE_S


def keypad_cut(speaker, active, keys: str) -> tuple:
    """Stop our speech for a key press. Returns (pending_remainder, queued, streaming)."""
    queued = len(speaker.queue)
    streaming = active is not None and active.is_alive()
    speaker.drop()
    remainder = resume_text_from(speaker.cut) if speaker.cut is not None else ''
    if streaming:
        active.cancel()          # stop queueing the rest of the reply
    clear_keys()
    return remainder, queued, streaming
REGISTRATION_TIMEOUT = 30
ANSWER_TIMEOUT = 90          # give up if unanswered
SILENCE_END_S = float(envc('VAD_SILENCE_S', '0.5'))   # silence that ends a turn
MAX_SPOKEN_CHUNK = int(envc('MAX_SPOKEN_CHUNK', '160'))   # flush a punctuation-less monologue
# How much of a reply may be SPOKEN. The transcript always keeps the whole thing; only the
# line is capped, so the owner is never talked over by a monologue he cannot interrupt.
MAX_SPOKEN_SENTENCES = int(envc('MAX_SPOKEN_SENTENCES', '3'))
MAX_SPOKEN_CHARS = int(envc('MAX_SPOKEN_CHARS', '160'))    # chat mode
# Explain mode: he asked to be told about something, so one chunk of roughly half a minute.
EXPLAIN_MAX_CHARS = int(envc('EXPLAIN_MAX_CHARS', '450'))
EXPLAIN_MAX_SENTENCES = int(envc('EXPLAIN_MAX_SENTENCES', '12'))
EXPLAIN_PATTERNS = envc('EXPLAIN_PATTERNS',
                        'tell me|tell us|explain|describe|talk about|talk me through|go on|'
                        'continue|keep talking|more about|in detail')
# What "go on" may mean, and what it must never mean. A resume needs an explicit instruction:
# a bare "okay" is an acknowledgement, not one -- on 15 October "Okay, stop." matched "ok" and
# resumed the reply the owner had just stopped.
RESUME_PATTERNS = envc('RESUME_PATTERNS',
                       r'(?:go on|continue|keep going|keep talking|carry on|more)')
# An answer to the offer "Shall I go on?" -- only while that offer is pending do these resume.
AFFIRM_PATTERNS = envc('AFFIRM_PATTERNS',
                       r'^(?:yes|yeah|yep|sure|please do|please continue|go ahead|'
                       r'ok(?:ay)?[, ]+(?:go on|continue|carry on))')
# Words that end a pending answer for good. They cancel; they never resume.
CANCEL_PATTERNS = envc('CANCEL_PATTERNS',
                       r'(?:^|\b)(?:stop|wait|no|nope|enough|quiet|hush|never mind|forget it|'
                       r'shut up)(?:\b|$)')
OFFER_TEXT = envc('OFFER_TEXT', 'Shall I go on?')
_EXPLAIN_RE = re.compile(EXPLAIN_PATTERNS, re.I)
_RESUME_RE = re.compile(RESUME_PATTERNS, re.I)
_AFFIRM_RE = re.compile(AFFIRM_PATTERNS, re.I)
_CANCEL_RE = re.compile(CANCEL_PATTERNS, re.I)
# Leading noise that does not change what a sentence means: "okay, go on" is still "go on".
_LEAD_FILLER_RE = re.compile(r'^(?:ok|okay|so|well|um|uh|er|and|but|right|alright)[,.\s]+', re.I)


def continuation_kind(text: str) -> tuple:
    """('resume'|'cancel'|'none', the phrase that decided it) for a turn that may answer the
    offer "Shall I go on?".

    Cancel wins outright, because the words that end an answer are the ones a half-heard
    sentence is most likely to contain ("Okay, stop.", "no thanks"). A bare "okay" is neither:
    it is an acknowledgement, so it does nothing at all. Only an explicit instruction --
    go on, continue, keep going, more, carry on, or yes while the offer is up -- resumes.
    """
    low = (text or '').strip().lower()
    if not low:
        return ('none', '')
    match = _CANCEL_RE.search(low)
    if match:
        return ('cancel', match.group(0).strip())
    core = _LEAD_FILLER_RE.sub('', low, count=1)
    match = _RESUME_RE.search(core) or _AFFIRM_RE.match(core)
    if match:
        return ('resume', match.group(0).strip())
    return ('none', '')
# A run-on sentence is broken at a clause boundary once it gets this long, so a reply is spoken
# in short natural phrases: the owner hears the first words sooner and the cap cuts between clauses.
CLAUSE_BREAK_CHARS = int(envc('CLAUSE_BREAK_CHARS', '70'))
STREAM_HERMES = envc('STREAM_HERMES', '1').lower() not in ('0', 'false', 'no', 'off')
INCOMPLETE_GUARD = envc('INCOMPLETE_GUARD', '1').lower() not in ('0', 'false', 'no', 'off')
REPLY_WAIT_S = float(envc('REPLY_WAIT_S', '8'))      # let an in-flight reply finish rather than
                                                      # abandoning it (abandoned work slows the next)
SHORT_UNPUNCTUATED_CHARS = int(envc('SHORT_UNPUNCTUATED_CHARS', '40'))
EXTENSION_SILENCE_S = float(envc('EXTENSION_SILENCE_S', '0.9'))  # wait this long before giving up
                                                                 # on a sentence he may finish
MAX_TURN_EXTENSIONS = int(envc('MAX_TURN_EXTENSIONS', '4'))
# Words that almost never end a finished spoken sentence: if the transcript stops on one
# without punctuation, the owner was mid-thought and the VAD heard a thinking pause.
DANGLING_WORDS = {'of', 'the', 'a', 'an', 'and', 'or', 'but', 'to', 'in', 'on', 'at', 'for',
                  'with', 'from', 'about', 'into', 'my', 'your', 'is', 'are', 'was', 'were',
                  'so', 'because', 'if', 'when', 'than', 'as', 'by', 'like', 'that'}
MIN_SPEECH_S = 0.3           # ignore shorter
IDLE_TIMEOUT = 60            # hang up after this long with no turn
TOTAL_TIMEOUT = 600          # hard cap on the call
SILENCE_WAV = str(pathlib.Path.home() / '.baresipy' / 'silence.wav')
# What the assistant calls itself on the line, and an optional folder of notes it may search
# read-only when the owner asks about his own things.
AGENT_NAME = envc('AGENT_NAME', 'Jarvis')
KNOWLEDGE_DIR = envc('KNOWLEDGE_DIR', '').strip()

_LOOKUP_ALWAYS = (
    " When I ask about me, my equipment, my machines or my projects, look it up before you "
    "answer: your own memory first. Say you do not know only when that comes back empty, and "
    "never tell me a thing was never mentioned without having looked."
)
_LOOKUP_NOTES = (
    " When I ask about me, my equipment, my machines or my projects, look it up before you "
    f"answer: your own memory first, then the notes under {KNOWLEDGE_DIR} -- searching and "
    "reading them is read-only and always allowed, and they are the record of what I have told "
    "you. Keep every lookup small: search rather than read whole files, take the matching lines "
    "with one line of context, and stop at about forty lines. Say you do not know only when both "
    "come back empty, and never tell me a thing was never mentioned without having looked."
)

SYSTEM_PROMPT = (
    "You are on a phone call with your owner. Your reply is spoken aloud by a "
    f"text-to-speech voice, so speak your own name as \"{AGENT_NAME}\" and never spell "
    "it out as letters. Match your length to the question: a chat "
    "remark or a simple fact gets one or two short sentences. When he asks you to tell, "
    "explain or describe something, lead with the point and keep going in short "
    "sentences for up to about thirty seconds, then stop and offer to go on. "
    "No markdown, lists, emoji, code or URLs. If something will take a while, say so briefly "
    "first."
    + (_LOOKUP_NOTES if KNOWLEDGE_DIR else _LOOKUP_ALWAYS)
    + " Anything I ask by voice may be misheard. Risky actions -- deleting or changing files, "
    "restarting services, spending money -- are put to me as a spoken approval question by the "
    "telephone system itself: go ahead and do the thing and say in the same breath what you are "
    "about to do, rather than asking me to confirm first, and never call it done until it is."
)
HANGUP_PHRASES = ('goodbye', 'bye bye', 'hang up', 'that is all')

_log_lock = threading.Lock()

# A call in progress, on disk. The gateway answers tool calls on a worker thread where the
# request-scoped "this is a call" marker is not visible, so the approval plugin could not tell a
# call from any other API client: a call once created a file in the owner's folder
# with no question asked. This file is the fact, not a hint: written when the call starts, with
# the pid, and read back only while that pid is alive.
LIVE_CALL_FILE = av.APPROVAL_DIR / 'live-call'
# What he last said, for trusted voice mode (item 5): the gateway decides there whether an
# action may run without a question, and it may only do that from HIS words. Written when the
# turn is understood, with what the transcription did.
TURN_FILE = av.APPROVAL_DIR / 'last-turn.json'
# Set by the transcription helpers when this turn needed a retry or produced two different texts.
STT_STATE = {'retried': False, 'dirty': ''}


def publish_turn(text: str, complete: bool, clean: bool = True, dirty_reason: str = '') -> None:
    """Record the owner's latest spoken turn where the gateway can read it (mode 600)."""
    try:
        TURN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TURN_FILE.write_text(json.dumps({'text': text, 'when': time.time(),
                                         'complete': bool(complete), 'clean': bool(clean),
                                         'dirty_reason': dirty_reason, 'source': 'voice'}))
        os.chmod(TURN_FILE, 0o600)
    except Exception:
        pass


def clear_turn() -> None:
    try:
        TURN_FILE.unlink()
    except OSError:
        pass


def mark_call_live() -> None:
    try:
        LIVE_CALL_FILE.parent.mkdir(parents=True, exist_ok=True)
        LIVE_CALL_FILE.write_text('%d %d\n' % (os.getpid(), int(time.time())))
        os.chmod(LIVE_CALL_FILE, 0o600)
    except OSError:
        pass


def clear_call_live() -> None:
    try:
        LIVE_CALL_FILE.unlink()
    except OSError:
        pass


def log(line: str) -> None:
    stamp = time.strftime('%H:%M:%S')
    text = f'{stamp} {line}'
    print(text, flush=True)
    with _log_lock:
        with open(LOG_PATH, 'a') as fh:
            fh.write(text + '\n')
    try:
        os.chmod(LOG_PATH, 0o600)
    except OSError:
        pass


# --- scheduled-call mode ------------------------------------------------------

def load_call_spec(path=None):
    """The prepared script for a scheduled call, or None on an ordinary call.

    Shape: {'kind', 'lines' [spoken, in order], 'question', 'reminder', 'context',
    'closing_idle_s', 'tts_slow', 'result', 'max_s'}. A malformed or missing file is reported
    and ignored rather than killing the call.

    The path comes from --script on the command line, else PHONE_SCRIPT_FILE. Both exist
    because call_trigger.sh forwards arguments and the older path set the variable; a spec
    that cannot be read must never be the reason a call does nothing silently.
    """
    path = path or SCRIPT_FILE
    if not path:
        return None
    try:
        with open(path) as fh:
            spec = json.load(fh)
    except Exception as exc:
        log(f'scheduled call: cannot read spec {path}: {type(exc).__name__}: {exc}')
        return None
    spec.setdefault('lines', [])
    spec.setdefault('question', '')
    spec.setdefault('reminder', '')
    spec.setdefault('context', '')
    spec.setdefault('closing_idle_s', 0)
    spec.setdefault('tts_slow', 1.0)
    spec.setdefault('listen_silence_s', 0)
    spec.setdefault('result', '')
    spec.setdefault('max_s', 0)
    log(f'scheduled call ({spec.get("kind")}): {len(spec["lines"])} script line(s)'
        + (', one question' if spec['question'] else '')
        + (', a reminder before hanging up' if spec['reminder'] else ''))
    return spec


def apply_spec_listen(spec) -> float:
    """Take the turn-end silence for this call from its spec. Returns the value in use.

    A call that invites a long answer — one made while he is driving, where he cannot type and
    the call is his only channel — waits longer before deciding he has finished. Driving he
    pauses to think mid-sentence, and at the usual half-second of silence the call ends the
    turn and answers half of what he said. His complaint, 30 October 2026: *"if i talk for 10
    seconds or more, i think that it is not reliable."*

    A spec without the field leaves the default alone, so an ordinary call is unchanged. A
    value that is not a number is ignored rather than failing the call.
    """
    global SILENCE_END_S, EXTENSION_SILENCE_S
    raw = (spec or {}).get('listen_silence_s')
    if not raw:
        return SILENCE_END_S
    try:
        value = float(raw)
    except (TypeError, ValueError):
        log(f'scheduled call: ignoring a bad listen_silence_s value {raw!r}')
        return SILENCE_END_S
    if value > SILENCE_END_S:
        SILENCE_END_S = value
        # The held-open turn uses its own, longer threshold; raise it too, or a turn already
        # extended would still be cut short at the old value.
        EXTENSION_SILENCE_S = max(EXTENSION_SILENCE_S, value)
        log(f'scheduled call: long-answer mode -- waiting {value:.1f}s of silence before '
            f'ending a turn (normal {envc("VAD_SILENCE_S", "0.5")}s)')
    return SILENCE_END_S


def apply_spec_pace(spec) -> float:
    """Take the speech pace for this call from its spec. Returns the pace in use.

    A spec without the field, or with a value that is not a number, leaves the pace at its
    default rather than failing the call: a wrong pace is a small annoyance, a call that
    does not happen at all is not.
    """
    global TTS_SLOW
    if spec and spec.get('tts_slow'):
        try:
            TTS_SLOW = float(spec['tts_slow'])
        except (TypeError, ValueError):
            log(f'scheduled call: ignoring a bad tts_slow value {spec["tts_slow"]!r}')
        else:
            if abs(TTS_SLOW - 1.0) > 1e-6:
                log(f'scheduled call: speaking {TTS_SLOW:.2f}x slower than normal')
    return TTS_SLOW


def write_result(spec, outcome: str, heard: list, extra: dict | None = None) -> None:
    """The outcome file the job reads: answered, answered_silent or not_answered."""
    if not spec or not spec.get('result'):
        return
    data = {'kind': spec.get('kind'), 'outcome': outcome, 'utterances': list(heard or []),
            'turns': len(heard or []), 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    data.update(extra or {})
    try:
        path = pathlib.Path(spec['result'])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.tmp')
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
        os.chmod(tmp, 0o600)
        tmp.replace(path)
        log(f'scheduled call: outcome {outcome} ({len(heard or [])} heard turn(s)) -> {path}')
    except Exception as exc:
        log(f'scheduled call: could not write the outcome file: {type(exc).__name__}: {exc}')


def ends_incomplete(text: str) -> bool:
    """True when a transcript looks cut off mid-thought.

    The VAD cannot tell a thinking pause from a finished sentence: a flat 0.5 s of silence
    split "I am just testing the speed of | Conversation." into two turns, and the first
    was answered as if it were a whole question. A sentence that stops on a function word
    with no terminal punctuation is far more likely unfinished than finished, so keep
    listening instead of replying.
    """
    t = (text or '').strip()
    if not t:
        return False
    if t.endswith(('...', '..', ' —', ' --', ' -')):
        return True                 # trailing off mid-thought, not a finished sentence
    if t[-1] in '.!?':
        return False
    if len(t) <= SHORT_UNPUNCTUATED_CHARS:
        return True        # "Can you tell me" -- he was still composing; do not answer yet
    words = re.findall(r"[a-z']+", t.lower())
    return bool(words) and words[-1] in DANGLING_WORDS


def rms_of(pcm16: bytes) -> float:
    n = len(pcm16) // 2
    if n == 0:
        return 0.0
    s = struct.unpack_from('<%dh' % n, pcm16)
    return (sum(x * x for x in s) / n) ** 0.5


# --- HTTP helpers -------------------------------------------------------------
def _or_headers() -> dict:
    return {'Authorization': 'Bearer ' + OPENROUTER_KEY, 'Content-Type': 'application/json'}


def _pace_filter(slow: float) -> list:
    """ffmpeg arguments that stretch a clip by `slow` (>1 = slower), pitch unchanged.

    atempo alone covers 0.5-2.0; a chain keeps it correct for anything more extreme, so a
    future 'half speed' request needs no new code.
    """
    if abs(slow - 1.0) < 1e-6:
        return []
    tempo, parts = 1.0 / slow, []
    while tempo < 0.5:
        parts.append('atempo=0.5')
        tempo /= 0.5
    while tempo > 2.0:
        parts.append('atempo=2.0')
        tempo /= 2.0
    parts.append(f'atempo={tempo:.6f}')
    return ['-filter:a', ','.join(parts)]


def tts_to_clip(text: str, tag: str) -> tuple:
    """Text -> 48 kHz stereo 16-bit wav (the call's stream format). Returns (path, seconds)."""
    t0 = time.time()
    r = requests.post('https://openrouter.ai/api/v1/audio/speech', headers=_or_headers(),
                      json={'model': TTS_MODEL, 'input': text, 'voice': TTS_VOICE,
                            'response_format': 'mp3'}, timeout=120)
    r.raise_for_status()
    CLIP_DIR.mkdir(exist_ok=True)
    mp3 = CLIP_DIR / f'{tag}.mp3'
    wav = CLIP_DIR / f'{tag}-48k.wav'
    mp3.write_bytes(r.content)
    # The pace is applied here, after the provider has spoken, so it costs nothing and does
    # not touch the model, the voice or the request. The clip is re-measured afterwards, which
    # is what the speaker's timing uses -- a slowed clip must not be cut off early.
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(mp3)]
                   + _pace_filter(TTS_SLOW) + ['-ar', '48000', '-ac', '2', str(wav)], check=True)
    dur = wav_duration(wav)
    if _pace_filter(TTS_SLOW):
        log(f'pace {TTS_SLOW:.2f}x -> {dur:.2f}s clip ({tag}, {len(text)} chars)')
    return str(wav), dur, time.time() - t0


def _stt_once(wav_bytes: bytes) -> tuple:
    """One transcription request. Returns (text, seconds, usage)."""
    t0 = time.time()
    r = requests.post('https://openrouter.ai/api/v1/audio/transcriptions', headers=_or_headers(),
                      json={'model': STT_MODEL, 'language': STT_LANG,
                            'input_audio': {'data': base64.b64encode(wav_bytes).decode(),
                                            'format': 'wav'}}, timeout=180)
    r.raise_for_status()
    d = r.json()
    usage = d.get('usage', {}) or {}
    return (d.get('text') or '').strip(), time.time() - t0, usage


def stt(wav_bytes: bytes) -> tuple:
    """Transcribe, resending the same audio if the first attempt is slow.

    The provider's tail is long and unpredictable -- the same clip has come back in 0.47 s and in
    18.03 s. So: if nothing has arrived after STT_RETRY_S, send the identical request again and take
    whichever answer comes back first. At most one retry. Returns (text, seconds, usage).
    """
    if STT_RETRY_S <= 0:
        return _stt_once(wav_bytes)
    t0 = time.time()
    lock = threading.Lock()
    state = {'winner': None, 'retry_at': None, 'times': {}}

    def attempt(name: str) -> None:
        try:
            text, secs, usage = _stt_once(wav_bytes)
        except Exception as exc:
            log(f'STT {name} attempt failed: {type(exc).__name__}: {exc}')
            return
        with lock:
            state['times'][name] = time.time() - t0
            if state['winner'] is None:
                state['winner'] = (text, secs, usage, name)
            else:
                other = state['winner'][3]
                log(f'STT {name} also returned after {state["times"][name]:.2f}s -- discarded, '
                    f'{other} had it at {state["times"].get(other, 0):.2f}s')
                STT_STATE['dirty'] = 'two different transcripts for the same turn'
                return
        log(f'STT answered by {name} after {state["times"][name]:.2f}s')

    first = threading.Thread(target=attempt, args=('first',), daemon=True)
    first.start()
    first.join(timeout=STT_RETRY_S)
    if first.is_alive():
        state['retry_at'] = time.time() - t0
        log(f'STT: no answer after {STT_RETRY_S}s -- resending the same audio')
        retry = threading.Thread(target=attempt, args=('retry',), daemon=True)
        retry.start()
    deadline = t0 + 180
    while time.time() < deadline:
        with lock:
            if state['winner'] is not None:
                text, secs, usage, name = state['winner']
                if name == 'retry' and state['retry_at'] is not None:
                    STT_STATE['retried'] = True
                    log(f'STT retry won at {state["times"].get("retry", 0):.2f}s '
                        f'(the first attempt had not answered after {STT_RETRY_S}s'
                        f'{", it later took " + format(state["times"]["first"], ".2f") + "s" if "first" in state["times"] else ""})')
                return text, secs, usage
        time.sleep(0.02)
    raise TimeoutError('no transcription answer within 180s')


def should_speak_hold(streamer, said: bool, suppressed: bool, now: float) -> bool:
    """Decide whether to fill a long silence with the holding line.

    Speak it only if a reply is being generated, nothing has been said yet, the wait has passed
    the threshold, and the owner has not started talking (in which case he is the one waiting).
    """
    if streamer is None or said or suppressed:
        return False
    if streamer.first_sentence_s is not None or streamer.speech_capped:
        return False
    if streamer.stopped_by_cancel or not streamer.is_alive():
        return False
    return now - streamer.started_at >= HOLD_LINE_S


def detect_mode(text: str) -> tuple:
    """Which reply mode his transcript calls for: ('explain'|'chat', the pattern that matched).

    Deliberately simple patterns, kept in the .env and logged on every turn, so a wrong mode is easy
    to spot and easy to tune without touching code.
    """
    m = _EXPLAIN_RE.search(text or '')
    return ('explain', m.group(0)) if m else ('chat', '')


def resume_text_from(cut: dict) -> str:
    """The text to speak next, taken from the clip that was interrupted mid-play.

    Everything after it was never heard, whether it was queued, generated but unspoken, or written
    after the interruption -- so the whole tail from this clip's first word is what "go on" owes him.
    """
    if not cut:
        return ''
    src = cut.get('reply')
    text = src.text if isinstance(src, StreamingReply) else (src or '')
    start = cut.get('start') or 0
    return (text[start:] if start else text).strip()


def mode_limits(mode: str) -> tuple:
    """(characters, sentences) that may be spoken in this mode."""
    if mode == 'explain':
        return EXPLAIN_MAX_CHARS, EXPLAIN_MAX_SENTENCES
    return MAX_SPOKEN_CHARS, MAX_SPOKEN_SENTENCES


def speak_chunk(text: str, tag: str, max_chars: int, speaker) -> str:
    """Speak a stretch of already-written reply in short clips; return what is left unspoken.

    Used by the explain-mode continuation, so 'go on' resumes where speech stopped without
    another model call.
    """
    remaining = (text or '').strip()
    source = text or ''
    spoken, n = 0, 0
    while remaining and spoken < max_chars:
        at = source.find(remaining[:30]) if remaining[:30] else 0
        sentence, rest = split_sentence(remaining, MAX_SPOKEN_CHUNK)
        if not sentence:
            sentence, rest = remaining, ''
        spoken += len(sentence)
        try:
            path, dur, secs = tts_to_clip(sentence, f'{tag}-s{n}')
            speaker.say(path, dur, {'start': max(at, 0), 'text': sentence, 'reply': source})
            n += 1
            log(f'    TTS {secs:.2f}s clip={dur:.2f}s  "{sentence[:70]}"')
        except Exception as exc:
            log(f'    TTS failed: {type(exc).__name__}: {exc}')
        remaining = rest.strip()
    return remaining


def accumulate_speech(speech_s: float, chunk_s: float, level: float) -> float:
    """Running total of how long he has been speaking, for the holding-line decision.

    A single blip above the noise gate must not count: the first version suppressed the holding
    line on any 0.1 s of line noise above VAD_RMS (800) against his voice at 3000+, so it never
    spoke at all in the 17:37 call.
    """
    if level >= VAD_RMS:
        return speech_s + chunk_s
    return max(0.0, speech_s - chunk_s)


def hold_stands_down(active, said: bool, suppressed: bool, speech_s: float) -> bool:
    """True when the holding line should stay silent because he is the one talking."""
    if suppressed:
        return True
    return active is not None and not said and speech_s >= HOLD_SUPPRESS_S


def hermes_body(messages: list, stream: bool) -> dict:
    """The request body for the gateway, with the phone-call overrides applied.

    `PHONE_MODEL` picks the model for a call and MUST travel with an explicit ``provider``:
    the gateway's OpenAI-compatible endpoint
    (``gateway/platforms/api_server.py::_request_agent_overrides``) honours a per-request
    ``model`` only when ``provider`` is present, because ``direct_model_requests`` defaults
    to off. A bare model is silently ignored and the session answers as the gateway default.

    With an OpenRouter preset the model id carries the preset as a suffix —
    ``<model>@preset/<slug>``, e.g. ``deepseek/deepseek-v4.1-flash@preset/phone-call``.
    Both halves are needed and neither is cosmetic:

    * OpenRouter applies the preset (its provider order, its fallbacks) because the
      suffix names it; the generation record shows the preset id.
    * The gateway must still recognise the model, because it decides whether to send a
      ``reasoning`` block at all from the model string
      (``agent/reasoning_params.py::_supports_reasoning_extra_body``, whose prefix list
      matches ``deepseek/…`` but never a bare ``@preset/…``). A bare preset id therefore
      gets NO reasoning field — whatever the caller asks for is dropped — and the model
      thinks by default, which is what made every call bill reasoning tokens.

    Reasoning is read by the gateway ONLY from ``model_options.reasoning``
    (``api_server._runtime_options_from_model_options``); the top-level ``reasoning`` and
    ``reasoning_effort`` fields this function used to send were ignored on every call and
    have been removed. Measured 15 October 2026: 40 of 40 calls at 0 reasoning tokens with
    ``model_options`` against ~94 tokens per call before.

    When PHONE_MODEL is empty the body carries the gateway's own model and no provider
    field, exactly as before.
    """
    body = {'model': HERMES_MODEL, 'messages': messages}
    if PHONE_MODEL:
        body['model'] = PHONE_MODEL
        body['provider'] = 'openrouter'
    if stream:
        body['stream'] = True
        body['stream_options'] = {'include_usage': True}
    if PHONE_REASONING:
        effort = str(PHONE_REASONING).strip().lower()
        if effort in ('', 'none', 'off', 'false', 'disabled'):
            body['model_options'] = {'reasoning': {'enabled': False}}
        else:
            body['model_options'] = {'reasoning': {'effort': effort}}
    return body


def hermes_chat(messages: list) -> tuple:
    t0 = time.time()
    r = requests.post(HERMES_URL, headers={'Authorization': 'Bearer ' + HERMES_KEY,
                                          'X-Hermes-Call': '1',
                                           'Content-Type': 'application/json'},
                      json=hermes_body(messages, stream=False),
                      timeout=HERMES_TIMEOUT)
    r.raise_for_status()
    d = r.json()
    return d['choices'][0]['message']['content'].strip(), time.time() - t0, d.get('usage', {}) or {}


_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


def split_sentence(buf: str, max_chars: int) -> tuple:
    """Peel the first speakable chunk off a growing reply.

    Speaking starts as soon as there is something worth saying, so the first words leave while
    the rest of the reply is still being written. A clause boundary inside a long sentence is
    preferred over the sentence as a whole: the model writes 150-200 character "sentences", and
    a whole one is 12 s of uninterrupted talking at the caller. A punctuation-less stretch is
    flushed at max_chars, so a monologue never leaves the caller in silence either.
    """
    m = _SENTENCE_END.search(buf)
    limit = m.end() if m else len(buf)
    if limit >= CLAUSE_BREAK_CHARS:
        cut = next((i for i, ch in enumerate(buf) if 30 <= i < limit and ch in ',;:'
                    and buf[i + 1:i + 2] in (' ', '')), None)
        if cut is not None:
            return buf[:cut + 1].strip(), buf[cut + 1:]
    if m:
        return buf[:limit].strip(), buf[limit:]
    if len(buf) >= max_chars:
        cut = buf.rfind(' ', 0, max_chars)
        cut = cut if cut > 20 else max_chars
        return buf[:cut].strip(), buf[cut:]
    return '', buf


class StreamingReply(threading.Thread):
    """Read a Hermes reply as it is generated and queue each finished sentence to speak.

    Runs in its own thread because the main loop must keep ticking the speech queue and
    watching for barge-in while the reply arrives. It only appends clips to the queue --
    every write to baresip's pty still happens in the main loop, through SpeechQueue.

    The stream carries reasoning_content deltas alongside content; only content is
    spoken, otherwise the caller would hear the model thinking out loud.
    """

    def __init__(self, speaker, messages: list, tag: str, turn: int, mode: str = 'chat',
                 max_chars: int = None, max_sentences: int = None):
        super().__init__(daemon=True)
        self.speaker, self.messages, self.tag, self.turn = speaker, messages, tag, turn
        self.mode = mode
        self.max_chars = int(max_chars or MAX_SPOKEN_CHARS)
        self.sentences_limit = int(max_sentences or MAX_SPOKEN_SENTENCES)
        self.spoken_chunks = []
        self.offer_handled = False
        self.text = ''
        self.error = ''
        self.usage = {}
        self.first_sentence_s = None
        self.total_s = None
        self.recorded = False
        self.cancelled = threading.Event()
        self.spoken_sentences = 0
        self.spoken_chars = 0
        self.speech_capped = False
        self.stopped_by_cancel = False
        self._buf = ''
        self._n = 0
        self._t0 = 0.0
        self.started_at = 0.0
        self._search_from = 0       # where to look for the next chunk's position in .text

    def unspoken(self) -> str:
        """What was written but not spoken, so 'go on' can continue exactly from there."""
        if not self.spoken_chunks or not self.text:
            return ''
        i = self.text.rfind(self.spoken_chunks[-1])
        return self.text[i + len(self.spoken_chunks[-1]):].strip() if i >= 0 else ''

    def cancel(self) -> None:
        """Owner interrupted: stop generating, stop queueing further speech."""
        self.cancelled.set()

    def _emit(self, chunk: str) -> None:
        if not chunk:
            return
        if self.speech_capped:
            return                      # keep reading for the transcript, just stop talking
        if self.spoken_chars and self.spoken_chars + len(chunk) > self.max_chars:
            self.speech_capped = True
            log(f'spoken cap reached at {self.spoken_sentences} phrase(s), '
                f'{self.spoken_chars} chars -- the rest of the reply stays in the transcript')
            return
        if not self.spoken_chars and len(chunk) > self.max_chars:
            cut = chunk.rfind(' ', 0, self.max_chars)
            chunk = chunk[:cut if cut > 40 else self.max_chars].rstrip(' ,;:') + '.'
        if self.first_sentence_s is None:
            self.first_sentence_s = time.time() - self._t0
        try:
            path, dur, secs = tts_to_clip(chunk, f'{self.tag}-s{self._n}')
            self._n += 1
            where = self.text.find(chunk, self._search_from)
            if where >= 0:
                self._search_from = where + len(chunk)
            else:
                where = self._search_from          # text and clips drifted; stay monotonic
            if not self.cancelled.is_set():
                self.speaker.say(path, dur, {'start': where, 'text': chunk, 'reply': self})
            log(f'TTS {secs:.2f}s clip={dur:.2f}s  "{chunk[:70]}"')
            self.spoken_sentences += 1
            self.spoken_chars += len(chunk)
            self.spoken_chunks.append(chunk)
            if self.spoken_sentences >= self.sentences_limit:
                self.speech_capped = True
                log(f'spoken cap reached at {self.spoken_sentences} phrases, '
                    f'{self.spoken_chars} chars -- the rest of the reply stays in the transcript')
        except Exception as exc:
            log(f'TTS failed: {type(exc).__name__}: {exc}')

    def _handle_approval(self, event: dict) -> None:
        """A gated action needs the owner's spoken yes. Ask through the same in-call path the
        plugin uses, then send the answer back. Anything unclear, silence or a timeout is a no."""
        run_id = str(event.get('run_id') or event.get('id') or '').strip()
        rid = f'in-call-{run_id or int(time.time())}'
        payload = {'command': event.get('command') or '', 'description': event.get('description') or ''}
        av.log_line(f'ASK (live call) {rid}')
        log(f'approval requested by the agent: {av.plain_action(payload)}')
        av.write_ask(rid, payload, inline=True)
        decision = 'deny'
        deadline = time.time() + APPROVAL_INCALL_WAIT_S
        while time.time() < deadline:
            if self.cancelled.is_set():
                break
            result = av.read_result(rid)
            if result:
                decision = 'yes' if result.get('decision') == 'yes' else 'deny'
                break
            time.sleep(0.25)
        av.clear_result(rid)
        av.log_line(f'REPLY {rid}: {decision}')
        self._send_approval_reply(run_id, decision)

    def _send_approval_reply(self, run_id: str, decision: str) -> None:
        if not run_id:
            log('approval reply skipped: no run id in the event')
            return
        base = HERMES_URL.rsplit('/v1/chat/completions', 1)[0]
        url = f'{base}/v1/runs/{run_id}/approval'
        try:
            r = requests.post(url, headers={'Authorization': 'Bearer ' + HERMES_KEY,
                                            'Content-Type': 'application/json'},
                              json={'choice': 'once' if decision == 'yes' else 'deny'},
                              timeout=10)
            log(f'approval reply for {run_id}: {decision} -> HTTP {r.status_code}')
        except Exception as exc:
            log(f'approval reply failed: {type(exc).__name__}: {exc}')

    def run(self) -> None:
        self._t0 = time.time()
        self.started_at = self._t0
        try:
            with requests.post(HERMES_URL,
                               headers={'Authorization': 'Bearer ' + HERMES_KEY,
                                        'Content-Type': 'application/json',
                                        'X-Hermes-Call': '1'},
                               json=hermes_body(self.messages, stream=True),
                               stream=True, timeout=HERMES_TIMEOUT) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if self.cancelled.is_set():
                        self.stopped_by_cancel = True
                        break
                    if not line or not line.startswith('data:'):
                        continue
                    payload = line[5:].strip()
                    if payload == '[DONE]':
                        break
                    try:
                        chunk = json.loads(payload)
                        if not isinstance(chunk, dict):
                            continue
                        first = chunk['choices'][0]
                        delta = first.get('delta') or {} if isinstance(first, dict) else {}
                    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                        # A non-dict where a chunk or a choice is expected must never kill the
                        # reply: that is how three replies were lost in the last call.
                        continue
                    if 'approval.request' in payload or chunk.get('type') == 'approval.request':
                        self._handle_approval(chunk)
                        continue
                    if chunk.get('usage'):
                        self.usage = chunk['usage']
                    piece = delta.get('content') or ''
                    if not piece:
                        continue                      # reasoning_content or role marker
                    self.text += piece
                    self._buf += piece
                    while True:
                        sentence, self._buf = split_sentence(self._buf, MAX_SPOKEN_CHUNK)
                        if not sentence:
                            break
                        self._emit(sentence)
            if self._buf.strip() and not self.cancelled.is_set():
                self._emit(self._buf.strip())
                self._buf = ''
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
        finally:
            self.total_s = time.time() - self._t0


def finish_stream(streamer) -> None:
    """Fold a streamed reply into the conversation once, and log what it cost in time."""
    if streamer is None or streamer.recorded:
        return
    streamer.recorded = True
    text = (streamer.text or '').strip()
    if text:
        streamer.messages.append({'role': 'assistant', 'content': text})
    first = ('n/a' if streamer.first_sentence_s is None
             else f'{streamer.first_sentence_s:.2f}s')
    # total_s stays None when the stream died before its first byte (a dropped call, a
    # provider error). Formatting it unguarded raised TypeError *inside finish_stream*,
    # which killed the driver before it wrote its outcome file -- so the call looked live
    # for the full 600 s wait and every offer behind it fell back to a text. Guard it.
    hermes_s = 'n/a' if streamer.total_s is None else f'{streamer.total_s:.2f}s'
    tok = (f'tokens {streamer.usage.get("prompt_tokens")}'
           f'+{streamer.usage.get("completion_tokens")}'
           if streamer.usage else 'tokens n/a')
    log(f'turn {streamer.turn}: reply "{text}" (Hermes {hermes_s}, first '
        f'sentence {first}, {tok}, streamed)')
    if text and streamer.speech_capped:
        total_sentences = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
        log(f'turn {streamer.turn}: mode={streamer.mode}, spoke {streamer.spoken_sentences} of '
            f'{total_sentences} sentences (limit {streamer.sentences_limit}); the rest was '
            f'transcript only')
    if streamer.stopped_by_cancel:
        log(f'turn {streamer.turn}: stream cancelled mid-generation after {len(streamer.text)} '
            f'characters, {streamer.spoken_sentences} phrase(s) spoken, no error')
    if not text:
        log(f'turn {streamer.turn}: reply cancelled before any speech'
            f'{" (stream error: " + streamer.error + ")" if streamer.error else ""}')


def wav_duration(path) -> float:
    import wave
    with wave.open(str(path), 'rb') as w:
        return w.getnframes() / float(w.getframerate() or 1)


def write_wav16k_mono(path, pcm16: bytes) -> None:
    import wave
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm16)


class Transcriber(threading.Thread):
    """Run transcription -- and its retry -- off the main loop.

    The loop has to keep reading audio: while it sat inside a synchronous transcription request it
    detected no turn and no interruption at all, however loud the interruption was (a replay of one
    call said the gate should have fired where the log showed nothing). Only HTTP happens here; the
    result is handed back to the loop, which stays the single writer to baresip's pty.
    """

    def __init__(self, wav_bytes: bytes, turn: int):
        super().__init__(daemon=True)
        self.wav_bytes = wav_bytes
        self.turn = turn
        self.text = ''
        self.stt_s = 0.0
        self.usage = {}
        self.error = ''
        self.done = threading.Event()

    def run(self) -> None:
        try:
            self.text, self.stt_s, self.usage = stt(self.wav_bytes)
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
        finally:
            self.done.set()


# --- speaking into the live call ---------------------------------------------
class SpeechQueue:
    """Ordered, non-overlapping clips played into the live call.

    Deliberately single-threaded: started and stopped from the main loop, so only
    one writer touches baresip's pty (baresipy's own reader-thread polling is
    serialised by _PTY_LOCK above).

    Uses baresip's mixausrc module rather than baresipy's send_audio: baresip 4.x
    stalls the runtime /ausrc switch send_audio relies on -- a transmit-dump check
    showed only ~0.3 s of a 2 s clip reaching the wire -- whereas mixausrc mixes a
    file into the live encoding stream and stops instantly for barge-in.
    """

    BARGIN_MIN_PLAY_S = 0.5   # ignore barge-in during a clip's first moments

    def __init__(self, sip):
        self.sip = sip
        self.queue = []                # [(path, duration, meta)]
        self.current = None            # (path, ends_at, started_at, meta)
        self.cut = None                # meta of the clip that was interrupted mid-play

    def say(self, wav_path: str, duration: float, meta: dict = None) -> None:
        """Queue a clip. `meta` carries where its text sits in the reply, so an interruption can
        be located in the text rather than guessed at from what was merely queued."""
        self.queue.append((wav_path, duration, meta))

    def playing(self) -> bool:
        return self.current is not None

    def idle(self) -> bool:
        return self.current is None and not self.queue

    def tick(self) -> None:
        """Advance playback. Call from the main loop only."""
        if not self.sip.call_established:
            self.current, self.queue = None, []
            return
        if self.current is None:
            if not self.queue:
                return
            path, duration, meta = self.queue.pop(0)
            # 0% to the mic stream, 100% to the clip: the caller hears only us
            self.sip.do_command(f'/mixausrc_enc_start aufile {path} 0 100')
            now = time.time()
            self.current = (path, now + duration, now, meta)
            return
        if time.time() >= self.current[1]:
            self.sip.do_command('/mixausrc_enc_stop')
            self.current = None
            self.cut = None            # the last clip was heard in full

    def barge_in_ready(self) -> bool:
        """True only once the current clip has played long enough that speech on the
        line is the owner talking, not our own audio echoing back through the phone."""
        if self.current is None:
            return False
        return time.time() - self.current[2] >= self.BARGIN_MIN_PLAY_S

    def drop(self) -> None:
        """Barge-in: cut the current clip and discard anything queued.

        The clip being played is the first one not fully heard, so its text is where the
        explanation resumes when he says 'go on'."""
        self.queue.clear()
        if self.current is not None:
            self.cut = self.current[3]
            try:
                self.sip.do_command('/mixausrc_enc_stop')
            except Exception:
                pass
            self.current = None

    def drain(self) -> None:
        while not self.idle():
            self.tick()
            time.sleep(0.05)

    def shutdown(self) -> None:
        self.queue.clear()
        if self.current is not None:
            try:
                self.sip.do_command('/mixausrc_enc_stop')
            except Exception:
                pass
            self.current = None


# --- call setup --------------------------------------------------------------
def normalise_target(target: str) -> str:
    """baresipy appends the gateway to a bare number; a full URI needs the sip: scheme."""
    t = target.strip()
    if '@' in t and not t.lower().startswith('sip:'):
        t = 'sip:' + t
    return t


def place_call():
    target = normalise_target(TARGET_SIP)
    b = CallSIP(SIP_USER, SIP_PASS, SIP_GATEWAY, transport=SIP_TRANSPORT, headless=True,
                record_rx=True, block=False,
                media_encryption=SIP_MEDIAENC or None,
                login_options=envc('SIP_LOGIN_OPTIONS') or None)
    deadline = time.time() + REGISTRATION_TIMEOUT
    while not b.ready and b.is_alive() and time.time() < deadline:
        time.sleep(0.5)
    if not b.ready:
        log(f'registration not complete after {REGISTRATION_TIMEOUT}s -- ready={b.ready} '
            f'alive={b.is_alive()} status={b.call_status}')
        b.quit()
        return None, None
    log('registered')
    b.call(target)
    log(f'dialling {target}')
    t0 = time.time()
    while not b.call_established and b.is_alive() and time.time() - t0 < ANSWER_TIMEOUT:
        time.sleep(0.2)
    if not b.call_established:
        log(f'no answer after {ANSWER_TIMEOUT}s -- status={b.call_status}')
        b.quit()
        return None, None
    log(f'answered after {time.time() - t0:.1f}s')
    return b, b.get_rx_stream()


def main() -> int:
    ap = argparse.ArgumentParser(description='Call TARGET_SIP and hold a spoken conversation.')
    ap.add_argument('opening', nargs='?', default=f'Hello, this is {AGENT_NAME} calling.')
    ap.add_argument('--script', dest='script_arg', default=None,
                    help='path to a scheduled-call spec (the same as PHONE_SCRIPT_FILE)')
    args = ap.parse_args()

    # A scheduled call runs from a prepared spec: it speaks the script, asks its question,
    # then behaves like any other call. Without one, everything below is as it was.
    spec = load_call_spec(args.script_arg)
    # The pace is per call: the morning briefing may ask for a slower reading, and every other
    # call -- the evening one, a call he asked for in chat -- keeps the normal pace.
    apply_spec_pace(spec)
    # The turn-end silence is per call too: a call made while he is driving waits longer, so a
    # long spoken answer is not cut off mid-sentence. Every other call is unchanged.
    apply_spec_listen(spec)
    heard = []                  # what he actually said, for the outcome file
    max_call_s = TOTAL_TIMEOUT
    if spec and spec['max_s']:
        max_call_s = float(spec['max_s'])
    reminder_said = False

    for name, val in (('SIP_USER', SIP_USER), ('SIP_PASS', SIP_PASS), ('TARGET_SIP', TARGET_SIP),
                      ('OPENROUTER_API_KEY', OPENROUTER_KEY), ('HERMES_API_KEY', HERMES_KEY)):
        if not val:
            log(f'missing required setting: {name}')
            write_result(spec, 'not_answered', heard, {'error': f'missing setting {name}'})
            return 2

    # Pre-synthesise what the call will say while the phone is still ringing: the first
    # synthesis in a process is the cold one (1.4-1.7 s, measured), and doing it here
    # takes it off the critical path -- the clips are ready the moment he answers.
    intro_clips = []
    if spec:
        for i, line in enumerate([l for l in spec['lines'] if l.strip()]):
            try:
                path, dur, secs = tts_to_clip(line, f'intro{i}')
                intro_clips.append((path, dur))
                log(f'intro {i} pre-synthesised in {secs:.2f}s ({dur:.2f}s)')
            except Exception as exc:
                log(f'intro {i} pre-synthesis failed: {type(exc).__name__}: {exc}')
        if spec['question']:
            try:
                path, dur, secs = tts_to_clip(spec['question'], 'question')
                intro_clips.append((path, dur))
                log(f'question pre-synthesised in {secs:.2f}s ({dur:.2f}s)')
            except Exception as exc:
                log(f'question pre-synthesis failed: {type(exc).__name__}: {exc}')
    else:
        opening_clip = None
        try:
            opening_clip = tts_to_clip(args.opening, 'opening')
            log(f'opening pre-synthesised in {opening_clip[2]:.2f}s ({opening_clip[1]:.2f}s clip)')
        except Exception as exc:
            log(f'opening pre-synthesis failed: {type(exc).__name__}: {exc}')
        if opening_clip:
            intro_clips.append((opening_clip[0], opening_clip[1]))

    b, stream = place_call()
    if b is None:
        write_result(spec, 'not_answered', heard, {'why': 'no answer or no registration'})
        return 1
    if stream is None:
        log('no rx stream')
        write_result(spec, 'not_answered', heard, {'why': 'no rx stream'})
        b.hang()
        b.quit()
        return 1

    rate, channels = stream.sample_rate, stream.channels
    chunk_bytes = int(rate * channels * 2 * 0.1) or 1920
    log(f'rx stream: rate={rate} channels={channels} chunk={chunk_bytes}B')

    speaker = SpeechQueue(b)

    messages = [{'role': 'system', 'content': SYSTEM_PROMPT + (
        '\n\n' + spec['context'] if spec and spec.get('context') else '')}]
    call_start = time.time()
    mark_call_live()
    last_turn = time.time()
    utterance = bytearray()
    speech_s = 0.0
    silence_s = 0.0
    turn = 0
    active = None
    pending_audio = b''
    pending_speech = 0.0
    pending_text = ''
    pending_remainder = ''      # written but not yet spoken, for 'go on'
    queued_turns = []           # utterances whose audio is complete, awaiting transcription
    transcriber = None          # transcription worker, so the audio loop never blocks on it
    extensions = 0
    barge_count = 0
    speaking_last = False       # F: was our own speech playing on the previous iteration
    speak_ended_at = 0.0        # ... and when it stopped, for the listen-again grace pause
    hold_line_said = False
    hold_suppressed = False
    hold_speech_s = 0.0
    barge_loud = []
    barge_candidate_logged = False
    barge_first_loud = 0.0
    approval = None             # the spoken approval in progress, if any
    stop_call = False           # set when the approval is settled and the call should end

    def speak(text: str, tag: str) -> None:
        try:
            path, dur, secs = tts_to_clip(text, tag)
            speaker.say(path, dur)
            log(f'TTS {secs:.2f}s clip={dur:.2f}s  "{text[:70]}"')
        except Exception as exc:
            log(f'TTS failed: {type(exc).__name__}: {exc}')

    if APPROVAL_ONLY:
        # This call exists to ask one question: it is armed the moment the question has
        # finished playing, and nothing said before that counts.
        approval = {'id': APPROVAL_ID, 'armed': False, 'reasked': False, 'deadline': 0.0,
                    'inline': False}
        log(f'approval-only call for {APPROVAL_ID}: "{APPROVAL_TEXT[:100]}"')
    if intro_clips:
        for path, dur in intro_clips:
            speaker.say(path, dur)
        log(f'{len(intro_clips)} pre-warmed clip(s) queued '
            f'({sum(d for _, d in intro_clips):.1f}s of speech)')
    elif spec:
        # Nothing pre-synthesised: say it live rather than going silent.
        for line in [l for l in spec['lines'] if l.strip()] + ([spec['question']] if spec['question'] else []):
            speak(line, 'intro-live')
    else:
        speak(args.opening, 'opening')

    try:
        while True:
            if not b.call_established:
                log('call ended')
                break
            if time.time() - call_start > max_call_s:
                log(f'total call timeout reached ({max_call_s:.0f}s)')
                break
            if time.time() - last_turn > IDLE_TIMEOUT:
                log('idle timeout reached (no turn)')
                break
            if stop_call:
                log('hanging up: the approval is settled')
                break

            # -- spoken approval: ask if asked, arm when the question has finished, listen --
            if approval is None and not APPROVAL_ONLY and speaker.idle():
                ask = av.take_ask()
                if ask:
                    approval = {'id': ask['id'], 'armed': False, 'reasked': False,
                                'deadline': 0.0, 'inline': True}
                    av.log_line(f'ASK (in call) {ask["id"]}')
                    log(f'approval ask arrived mid-call: {ask["id"]}')
                    speak(ask.get('text') or av.question_text(ask.get('payload') or {}, inline=True),
                          f'ask-{ask["id"]}')
            if approval is not None:
                if av.is_cancelled(approval['id']):
                    av.log_line(f'CANCELLED before an answer: {approval["id"]}')
                    av.clear_cancel(approval['id'])
                    log(f'approval {approval["id"]}: settled on another surface; not asking')
                    if APPROVAL_ONLY:
                        speak('That was answered elsewhere, sir.', 'cancelled')
                        stop_call = True
                    approval = None
                elif not approval['armed'] and speaker.idle():
                    # The question has finished. Anything captured while it was playing is
                    # not an answer, so it is discarded here rather than judged.
                    if speech_s > 0:
                        log(f'  discarded {speech_s:.1f}s spoken before the question finished')
                    utterance, speech_s, silence_s = bytearray(), 0.0, 0.0
                    pending_audio, pending_speech = b'', 0.0
                    approval['armed'] = True
                    approval['deadline'] = time.time() + av.ANSWER_WINDOW_S
                    log(f'approval {approval["id"]}: question done, listening '
                        f'{av.ANSWER_WINDOW_S:.0f}s for approve or deny')
                elif approval['armed'] and time.time() > approval['deadline']:
                    av.log_line(f'REFUSED {approval["id"]} (silence)')
                    log(f'approval {approval["id"]}: no answer in '
                        f'{av.ANSWER_WINDOW_S:.0f}s -- treated as no')
                    av.write_result(approval['id'], 'no', [], {'source': 'silence'})
                    speak(av.ack_text('no'), 'ack-silence')
                    if APPROVAL_ONLY:
                        speaker.drain()
                        stop_call = True
                    approval = None

            # -- a scheduled call closes itself ------------------------------------
            # Once the script and its question are done and he has had his say, silence
            # ends the call rather than leaving it open until the idle timeout.
            if (spec and spec['closing_idle_s'] and heard and speaker.idle()
                    and not queued_turns and transcriber is None and not pending_audio
                    and time.time() - last_turn > spec['closing_idle_s']):
                log(f'{spec.get("kind")} call: {spec["closing_idle_s"]:.0f}s of silence after '
                    f'{len(heard)} turn(s) -- closing')
                if spec['reminder'] and not reminder_said:
                    reminder_said = True
                    speak(spec['reminder'], 'reminder')
                    speaker.drain()
                    time.sleep(0.7)
                speak(CLOSE_LINE, 'close')
                speaker.drain()
                time.sleep(0.7)
                break

            # -- transcription worker: start one, consume it when it lands -------------
            if transcriber is None and queued_turns and not pending_audio:
                head = queued_turns[0]
                transcriber = Transcriber(head['wav'], head['turn'])
                transcriber.start()
                log(f'turn {head["turn"]}: transcribing off the loop'
                    + (f' ({len(queued_turns)} queued)' if len(queued_turns) > 1 else ''))
            if transcriber is not None and transcriber.done.is_set():   # Event() is always truthy
                item = queued_turns.pop(0)
                tr, transcriber = transcriber, None
                turn = item['turn']
                if tr.error:
                    log(f'turn {turn}: STT failed: {tr.error}')
                    speak('Sorry, something went wrong.', f'err-{turn}')
                else:
                    text, stt_s, usage = tr.text, tr.stt_s, tr.usage
                    if len(text) < 2:
                        log(f'turn {turn}: STT empty ({stt_s:.2f}s)')
                    else:
                        log(f'turn {turn}: heard "{text}" (STT {stt_s:.2f}s cost '
                            f'{usage.get("cost")}; off the loop)')
                        heard.append(text)
                        if approval is not None:
                            # An approval answer is a control word, not conversation: it is
                            # judged here and never reaches the agent or the transcript.
                            if not approval['armed']:
                                log(f'turn {turn}: ignored -- said before the approval '
                                    f'question had finished playing')
                                continue
                            verdict = av.classify(text)
                            av.log_line(f'answer ({verdict}) {approval["id"]}: "{text[:80]}"')
                            if verdict == 'unknown' and not approval['reasked']:
                                approval['reasked'] = True
                                approval['deadline'] = time.time() + av.ANSWER_WINDOW_S
                                log(f'turn {turn}: neither approve nor deny; asking once more')
                                speak(av.reask_text(), f'reask-{turn}')
                                continue
                            decision = 'yes' if verdict == 'yes' else 'no'
                            av.write_result(approval['id'], decision, [text],
                                            {'source': 'call' if APPROVAL_ONLY else 'in-call',
                                             'how': verdict})
                            av.log_line(f'{"APPROVED" if decision == "yes" else "REFUSED"} '
                                        f'{approval["id"]} ({verdict})')
                            log(f'turn {turn}: approval {approval["id"]} -> {decision}')
                            speak(av.ack_text(decision), f'ack-{turn}')
                            if APPROVAL_ONLY:
                                speaker.drain()
                                stop_call = True
                            approval = None
                            continue
                        low = text.lower()
                        if any(ph in low for ph in HANGUP_PHRASES):
                            if spec and spec['reminder'] and not reminder_said:
                                reminder_said = True
                                speak(spec['reminder'], 'reminder')
                                speaker.drain()
                                time.sleep(0.7)
                            speak('Goodbye', f'bye-{turn}')
                            speaker.drain()      # let the farewell finish playing
                            time.sleep(4)
                            log('hanging up (farewell)')
                            break
                        kind, matched = (continuation_kind(text) if pending_remainder
                                         else ('none', ''))
                        if kind == 'cancel':
                            log(f'turn {turn}: mode=cancel (matched "{matched}") -- the pending '
                                f'answer is dropped for good ({len(pending_remainder)} characters)')
                            pending_remainder = ''
                            last_turn = time.time()
                        elif kind == 'resume':
                            log(f'turn {turn}: mode=continue (matched "{matched}"), no model call; '
                                f'{len(pending_remainder)} characters still to speak')
                            pending_remainder = speak_chunk(pending_remainder, f'cont-{turn}',
                                                            EXPLAIN_MAX_CHARS, speaker)
                            if pending_remainder:
                                speak(OFFER_TEXT, f'offer-{turn}')
                            else:
                                log(f'turn {turn}: the pending answer is finished')
                            last_turn = time.time()
                        elif (INCOMPLETE_GUARD and extensions < MAX_TURN_EXTENSIONS
                              and ends_incomplete(text)):
                            pending_audio, pending_speech, pending_text = (
                                item['clip'], item['speech_total'], text)
                            # Anything already queued is the rest of the same thought: merge it
                            # rather than answering the fragment on its own.
                            while queued_turns:
                                nxt = queued_turns.pop(0)
                                pending_audio += nxt['clip']
                                pending_speech += nxt['speech_total']
                                log(f'    merging turn {nxt["turn"]} into the held-open one')
                            speech_s, silence_s = 0.001, 0.0
                            extensions += 1
                            last_turn = time.time()
                            publish_turn(text, complete=False,
                                         clean=not STT_STATE['retried'],
                                         dirty_reason=STT_STATE['dirty'])
                            log(f'turn {turn}: heard "...{text[-45:]}" -- no terminal punctuation '
                                f'and it ends on "{text.split()[-1]}", so it reads as unfinished: '
                                f'staying on the line instead of answering')
                        else:
                            extensions = 0
                            if APPROVAL_ONLY:
                                log(f'turn {turn}: approval-only call -- not sent to the agent')
                                continue
                            if active is not None:
                                # An in-flight reply is NOT cancelled because a new turn arrived:
                                # an abandoned generation keeps running at the gateway and queues
                                # ahead of the next request. Wait for it (it speaks as it arrives);
                                # only a real barge-in cancels, in the barge-in branch below.
                                active.join(timeout=REPLY_WAIT_S)
                                if active.is_alive():
                                    log(f'turn {active.turn}: still generating after '
                                        f'{REPLY_WAIT_S}s, cancelled so the conversation can move on')
                                    active.cancel()
                                    active.join(timeout=3)
                                finish_stream(active)
                                active = None
                            if len(messages) > 21:          # system message + last 20
                                del messages[1:-20]
                            messages.append({'role': 'user', 'content': text})
                            publish_turn(text, complete=True,
                                         clean=not STT_STATE['retried'],
                                         dirty_reason=STT_STATE['dirty'])
                            STT_STATE['retried'] = False
                            STT_STATE['dirty'] = ''
                            mode, matched = detect_mode(text)
                            limit, limit_sentences = mode_limits(mode)
                            pending_remainder = ''      # a new question abandons any pending one
                            log(f'turn {turn}: mode={mode}'
                                + (f' (matched "{matched}")' if matched else '')
                                + f', speaking up to {limit} characters')
                            if STREAM_HERMES:
                                active = StreamingReply(speaker, messages, f'turn-{turn}', turn,
                                                        mode=mode, max_chars=limit,
                                                        max_sentences=limit_sentences)
                                active.start()
                                hold_line_said = hold_suppressed = False
                                hold_speech_s = 0.0
                            else:
                                try:
                                    reply, hermes_s, h_usage = hermes_chat(messages)
                                except Exception as exc:
                                    log(f'turn {turn}: Hermes failed: {type(exc).__name__}: {exc}')
                                    speak('Sorry, something went wrong.', f'err-{turn}')
                                else:
                                    messages.append({'role': 'assistant', 'content': reply})
                                    log(f'turn {turn}: reply "{reply}" (Hermes {hermes_s:.2f}s '
                                        f'tokens {h_usage.get("prompt_tokens")}'
                                        f'+{h_usage.get("completion_tokens")})')
                                    speak(reply, f'turn-{turn}')

            speaker.tick()                 # start/stop clips; single writer to the pty
            if not speaker.playing():
                if speaking_last:
                    speak_ended_at = time.time()      # our own speech just finished
                speaking_last = False
                barge_candidate_logged = False   # one 'candidate' note per clip at most
            else:
                speaking_last = True
            # F: with the keypad as the only interruption, a key is checked before the audio
            # level -- a key press makes no sound on the line.
            if KEYPAD_INTERRUPT:
                if approval is not None:
                    clear_keys()                 # a question is heard in full; keys are no answer
                elif speaker.playing():
                    stop_now, key_ms, keys = keypad_stop(
                        speaker.current[2] if speaker.current else 0.0, time.time(),
                        speaking=speaker.playing(), approval_active=False)
                    if stop_now:
                        remainder, queued, streaming = keypad_cut(speaker, active, keys)
                        if remainder:
                            pending_remainder = remainder
                            log(f'    keypad cut "{speaker.cut["text"][:40]}..." -> "go on" '
                                f'will resume there ({len(remainder)} characters)')
                        barge_count += 1
                        late = ' (LATE, over %.0f ms)' % (KEY_STOP_MAX_S * 1000) \
                            if key_ms is not None and key_ms > KEY_STOP_MAX_S * 1000 else ''
                        log(f'keypad-in #{barge_count}: key "{keys}" stopped the speech '
                            f'{key_ms} ms after it was pressed{late}; {queued} queued clip(s) '
                            f'dropped; stream {"cancelled" if streaming else "already finished"}')
            if active is not None and not active.is_alive() and not active.offer_handled:
                active.offer_handled = True
                if active.mode == 'explain' and active.speech_capped:
                    rest = active.unspoken()
                    if rest:
                        pending_remainder = rest
                        log(f'turn {active.turn}: explain chunk ended with {len(rest)} characters '
                            f'left; offering "{OFFER_TEXT}"')
                        speak(OFFER_TEXT, f'offer-{active.turn}')
            if should_speak_hold(active, hold_line_said, hold_suppressed, time.time()):
                hold_line_said = True
                log(f'no first sentence after {HOLD_LINE_S}s; saying "{HOLD_LINE_TEXT}"')
                speak(HOLD_LINE_TEXT, 'holding')
            raw = stream.read(chunk_bytes, timeout=1.0)
            if not raw:
                continue
            mono16 = resample_pcm16(raw, rate, channels, 16000)
            level = rms_of(mono16)
            chunk_s = len(raw) / float(rate * channels * 2)

            # F: with the keypad as the only interruption, whatever is said while we speak is
            # dropped here, so it can never come back later as a turn of its own.
            if KEYPAD_INTERRUPT and speaker.playing():
                continue
            # ... and our own speech just ended: listen again after a short pause.
            if KEYPAD_INTERRUPT and level >= VAD_RMS and speak_grace_active(speak_ended_at,
                                                                           time.time()):
                continue

            if level >= VAD_RMS:
                # An approval question is never interrupted: it must be heard in full, and
                # anything said over it is discarded rather than treated as an answer. Without
                # this, his first word cut the question off mid-sentence.
                if speaker.playing() and speaker.barge_in_ready() and approval is None \
                        and not KEYPAD_INTERRUPT:
                    # A marginal blip just above VAD_RMS is room noise, not the owner:
                    # require a sustained, clearly louder signal before cutting a reply.
                    now = time.time()
                    if level >= BARGIN_RMS:
                        if not barge_loud:
                            barge_first_loud = now     # the stop is measured from here
                        barge_loud.append(chunk_s)
                        if not barge_candidate_logged:
                            barge_candidate_logged = True
                            log(f'barge candidate: rms {level:.0f} ({sum(barge_loud):.1f}s of his '
                                f'audio above the gate so far)')
                    while len(barge_loud) > 1 and sum(barge_loud) > BARGIN_WINDOW_S:
                        barge_loud.pop(0)
                    if barge_loud and sum(barge_loud) >= BARGE_MIN_S - 0.02:
                        queued = len(speaker.queue)
                        streaming = active is not None and active.is_alive()
                        t_loud = barge_first_loud or time.time()
                        loud_seconds = sum(barge_loud)      # capture before the list is cleared
                        speaker.drop()
                        if speaker.cut is not None:
                            # He heard part of this clip, so "go on" owes him the whole of it and
                            # everything after -- not just whatever was still in the queue.
                            pending_remainder = resume_text_from(speaker.cut)
                            log(f'    interruption cut "{speaker.cut["text"][:40]}..." '
                                f'-> "go on" will resume there ({len(pending_remainder)} characters '
                                f'from character {speaker.cut.get("start")})')
                        if streaming:
                            active.cancel()      # stop queueing the rest of the reply
                        barge_count += 1
                        barge_loud = []
                        barge_candidate_logged = False
                        log(f'barge-in #{barge_count}: {loud_seconds:.2f}s of his audio above '
                            f'{BARGIN_RMS} (rms {level:.0f}); speech stopped '
                            f'{time.time() - t_loud:.3f}s after the first loud sample; '
                            f'{queued} queued clip(s) dropped; stream '
                            f'{"cancelled" if streaming else "already finished"}')
                else:
                    barge_loud = []
                hold_speech_s = accumulate_speech(hold_speech_s, chunk_s, level)
                if hold_stands_down(active, hold_line_said, hold_suppressed, hold_speech_s):
                    hold_suppressed = True       # he is genuinely talking; do not add a holding line
                utterance += mono16
                speech_s += 0.1
                silence_s = 0.0
                continue

            hold_speech_s = accumulate_speech(hold_speech_s, chunk_s, level)

            if speech_s > 0:
                silence_s += 0.1
                utterance += mono16
                if silence_s < (EXTENSION_SILENCE_S if pending_audio else SILENCE_END_S):
                    continue
                turn += 1
                clip = pending_audio + bytes(utterance)
                speech_total = speech_s + pending_speech
                pending_audio, pending_speech, pending_text = b'', 0.0, ''
                utterance, speech_s, silence_s = bytearray(), 0.0, 0.0
                last_turn = time.time()
                if speech_total < MIN_SPEECH_S:
                    log(f'turn {turn}: ignored, speech {speech_total:.1f}s')
                    continue
                CLIP_DIR.mkdir(exist_ok=True)
                wav_path = CLIP_DIR / f'turn-{turn}.wav'
                write_wav16k_mono(wav_path, clip)
                queued_turns.append({'turn': turn, 'clip': clip, 'speech_total': speech_total,
                                     'wav': wav_path.read_bytes()})
                log(f'turn {turn}: utterance ready ({speech_total:.1f}s of speech); '
                    f'{len(queued_turns)} awaiting transcription')
                continue

    except KeyboardInterrupt:
        log('interrupted')
    finally:
        clear_call_live()
        clear_turn()
        if approval is not None:
            # He hung up, or the line dropped, while the question was pending. Refuse it now
            # rather than leaving the gate to wait out its whole timeout.
            av.log_line(f'DENIED {approval["id"]} (the call ended before an answer)')
            log(f'approval {approval["id"]}: call ended with no answer -- sending deny')
            av.write_result(approval['id'], 'no', [], {'source': 'hangup'})
            approval = None
        if active is not None:
            active.cancel()
            active.join(timeout=5)
            finish_stream(active)
        if spec and spec['reminder'] and not reminder_said:
            # The reminder is owed on every way out that we still control: he said
            # goodbye (handled above), or the call is ending on a timeout.
            try:
                if b.call_established:
                    reminder_said = True
                    speak(spec['reminder'], 'reminder-close')
                    speaker.drain()
                    time.sleep(0.7)
            except Exception as exc:
                log(f'reminder could not be said: {type(exc).__name__}: {exc}')
        if spec:
            write_result(spec, 'answered' if heard else 'answered_silent', heard,
                         {'duration_s': round(time.time() - call_start, 1),
                          'closed_by': 'call'})
        speaker.shutdown()
        try:
            b.hang()
        except Exception:
            pass
        time.sleep(0.5)
        b.quit()
        log(f'call finished: {turn} turns, {barge_count} barge-ins, '
            f'{time.time() - call_start:.0f}s total')
    return 0


if __name__ == '__main__':
    sys.exit(main())
