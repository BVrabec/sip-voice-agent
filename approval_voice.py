#!/usr/bin/env python3
"""Shared pieces of the spoken approval: how it is worded, how an answer is judged,
and the file handshake between the gateway plugin and the call.

Three callers use it, and they must behave identically:
  * the `phone-approvals` plugin  -- owns the pending request and the timers
  * call_me.py in approval mode   -- a call made only to ask
  * call_me.py during a live call -- asked inside the call that is already up
Stdlib only: the gateway plugin imports this alongside Hermes, the call scripts
import it without Hermes at all.
"""
import hashlib
import json
import os
import pathlib
import re
import time

APPROVAL_DIR = pathlib.Path(os.environ.get('PHONE_APPROVAL_DIR')
                            or (pathlib.Path.home() / '.phone-agent' / 'approvals'))
ASK_DIR = APPROVAL_DIR / 'ask'            # plugin  -> call: ask this
RESULT_DIR = APPROVAL_DIR / 'results'     # call    -> plugin: he said this
CANCEL_DIR = APPROVAL_DIR / 'cancelled'   # plugin  -> call: never mind, it is settled
LOG_PATH = pathlib.Path(os.environ.get('PHONE_APPROVAL_LOG')
                         or (pathlib.Path.home() / '.phone-agent' / 'approval_calls.log'))

# How long we listen for an answer once the question has finished playing.
ANSWER_WINDOW_S = float(os.environ.get('PHONE_APPROVAL_ANSWER_S') or 20)

# 'approve' and 'approved' are the only words that mean yes. Everything else that is
# a clear refusal means no, and anything else is not an answer at all.
#
# Widened on 30 October 2026 at his request, after the transcription heard *"Approv."* and
# *"Aprova."* on two separate calls and logged both as refusals, so the gated action did not
# run. The pattern is deliberately narrow: it accepts his own pronunciation of the word
# (a-p-r-o-v-, with the second p dropped, and the -a ending Slovenian puts on it) and nothing
# else. `ap+rov` covers both spellings; `_NEG_RE` is still tested first, so "I do not approve"
# remains a no.
_YES_RE = re.compile(r'(?<!dis)\bap+rov(e|ed|al|a|ing)?\b', re.I)
_NO_RE = re.compile(r'\b(deny|denied|no|nope|refuse|refused|reject|rejected|negative|cancel|'
                    r'cancelled|canceled|stop|forget it|do not|don\'t|dont)\b', re.I)
# A denial anywhere in the sentence outranks the word 'approve' inside it: "I do not
# approve" must never be read as consent.
_NEG_RE = re.compile(r"\b(do not|don't|dont|does not|doesn't|did not|didn't|not|never|"
                     r"cannot|can't|won't|will not|would not|isn't|aren't)\b", re.I)

_SECRETISH_RE = re.compile(
    r'(sk-[A-Za-z0-9_-]{8,}|Bearer\s+\S+|\b[A-Za-z0-9_]*'
    r'(?:KEY|TOKEN|PASS|PASSWORD|SECRET|AUTH)[A-Za-z0-9_]*\s*[=:]\s*\S+|'
    r'\b[A-Fa-f0-9]{32,}\b|\b[A-Za-z0-9+/]{40,}={0,2}\b)', re.I)
_PATH_RE = re.compile(r'(/[^\s\'"]{2,120})')
_SERVICE_RE = re.compile(r'\b(?:systemctl|service)\b[^\n;|&]{0,40}?\b(?:restart|stop|start|reload|'
                         r'disable|enable)\s+([A-Za-z0-9_.@-]+)', re.I)
_CONTAINER_RE = re.compile(r'\b(?:docker|podman)\s+(?:compose\s+)?(restart|stop|kill|down|rm)\b\s*'
                           r'([A-Za-z0-9_.-]+)?', re.I)
_FILEISH_RE = re.compile(r'\b([A-Za-z0-9_.-]+\.(?:py|sh|json|md|yaml|yml|db|wav|txt|log|conf|env|'
                         r'csv|toml|ini|sqlite|sqlite3))\b', re.I)


def log_line(text: str) -> None:
    """Append one line to the approval log (mode 600). Never carries a secret: every
    caller sanitises first, and this scrubs again on the way out."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime('%H:%M:%S')
        with LOG_PATH.open('a') as fh:
            fh.write(f'{stamp} {_scrub(text)}\n')
        try:
            os.chmod(LOG_PATH, 0o600)
        except OSError:
            pass
    except Exception:
        pass


def _scrub(text: str) -> str:
    return _SECRETISH_RE.sub('<value withheld>', str(text or ''))


def classify(text: str) -> str:
    """'yes', 'no' or 'unknown'. Only a clear approve counts as yes."""
    t = re.sub(r'\s+', ' ', (text or '').strip().lower())
    if not t:
        return 'unknown'
    if _NEG_RE.search(t):
        return 'no'
    if _YES_RE.search(t):
        return 'yes'
    if _NO_RE.search(t):
        return 'no'
    return 'unknown'


# How a path is said out loud: the name, and where it lives. Override or extend with
# PHONE_PLACE_MAP="/srv/notes=in the notes, /mnt/data=on the data disk" (comma-separated
# prefix=phrase pairs); the built-in ones cover the home folder, /tmp and /etc.
def _place_prefixes():
    pairs = [(str(pathlib.Path.home()), 'in your home folder'), ('/tmp', 'in the temporary folder'),
             ('/etc', 'in /etc')]
    for item in (os.environ.get('PHONE_PLACE_MAP') or '').split(','):
        if '=' in item:
            prefix, phrase = item.split('=', 1)
            pairs.insert(0, (prefix.strip(), phrase.strip()))
    return tuple(pairs)


_PLACE_PREFIXES = _place_prefixes()


def _short_place(path: str) -> str:
    """A path in words: the name and where it lives, never the whole path. A spoken question
    that reads out /home/you/notes/test2.txt takes thirteen seconds; "test2.txt in your notes"
    takes four. A bare name stays a bare name."""
    p = str(path or '').strip().rstrip('.;,')
    name = os.path.basename(p.rstrip('/')) or p
    if not p.startswith('/'):
        return name
    for prefix, where in _PLACE_PREFIXES:
        if p == prefix or p.startswith(prefix + '/'):
            return f'{name} {where}'
    return f'{name} at {p}' if len(p) <= 28 else name


_VERB_RULES = (
    (re.compile(r'\b(rm|remove|delete|unlink|shred|trash)\b', re.I), 'Delete'),
    (re.compile(r'\b(install|apt|apt-get|dpkg|pip|npm)\b', re.I), 'Install'),
    (re.compile(r'\b(restart|reload|reboot)\b', re.I), 'Restart'),
    (re.compile(r'\b(write|create|touch|mkdir|save|new file)\b', re.I), 'Create'),
    (re.compile(r'\b(patch|edit|change|modify|update|replace|append)\b', re.I), 'Change'),
    (re.compile(r'\b(move|rename|mv)\b', re.I), 'Move'),
    (re.compile(r'\b(copy|cp)\b', re.I), 'Copy'),
    (re.compile(r'\b(stop|kill|down|disable)\b', re.I), 'Stop'),
    (re.compile(r'\b(push|commit|publish|send|upload|post)\b', re.I), 'Publish'),
    (re.compile(r'\b(run|execute|invoke|command)\b', re.I), 'Run'),
)
# What a command is *for*, in a few words, when it is not a plain file action. "Run X" tells the
# owner nothing; "Run a command that uses git in your second brain" does.
_PURPOSE_RULES = (
    (re.compile(r'\bgit\b', re.I), 'uses git'),
    (re.compile(r'\b(apt|apt-get|dpkg|pip|pip3|npm)\b', re.I), 'installs software'),
    (re.compile(r'\b(python|python3|node|perl|ruby|bash|sh)\b', re.I), 'runs a script'),
    (re.compile(r'\b(systemctl|service|journalctl)\b', re.I), 'changes a service'),
    (re.compile(r'\b(docker|podman)\b', re.I), 'changes a container'),
    (re.compile(r'\b(curl|wget|httpx|requests)\b', re.I), 'fetches something from the network'),
    (re.compile(r'\b(chmod|chown|truncate|dd|tee|tar|zip|unzip|mv|cp)\b', re.I), 'changes files'),
    (re.compile(r'\b(kill|pkill|reboot|shutdown)\b', re.I), 'stops something'),
    (re.compile(r'\b(sqlite3|mysql|psql)\b', re.I), 'queries a database'),
)
# Commands that act on one named file and are best described as "Delete/Create/Change <file>".
FILE_COMMANDS = frozenset({'rm', 'cp', 'mv', 'mkdir', 'rmdir', 'touch', 'truncate', 'chmod',
                           'chown', 'ln', 'dd', 'tee', 'install', 'unlink', 'shred'})
# The place, when the description names it in words instead of giving a path.
_PLACE_HINTS = (('home folder', ' in your home folder'),
                ('temporary folder', ' in the temporary folder'), ('/tmp', ' in the temporary folder'),
                ('/etc', ' in /etc'))
# Stray punctuation a description may carry ("Delete test2.txt; in your personal folder").
_STRAY_RE = re.compile(r'[;:,]+\s*|\s+[;:,]+')
_LEADING_VERB_RE = re.compile(
    r'^(delete|remove|create|write|make|change|modify|edit|move|copy|rename|install|update|'
    r'restart|stop|start|run|execute)\b[\s:;,-]*', re.I)


def _purpose(blob: str, target: str) -> str:
    for regex, phrase in _PURPOSE_RULES:
        if regex.search(blob):
            return f'{phrase}{" in " + target.split(" in ", 1)[1] if " in " in target else ""}'
    return f'runs {blob.split()[0]}' if blob.split() else 'changes something'


def _action_verb(blob: str) -> str:
    for regex, verb in _VERB_RULES:
        if regex.search(blob or ''):
            return verb
    return ''


def plain_action(payload: dict) -> str:
    """One short phrase naming the action and its target — "Delete test2.txt in your personal
    folder", or, when it is not a plain file action, "Run a command that uses git in your second
    brain". Never a long command, never a whole path, never anything credential-shaped."""
    desc = _STRAY_RE.sub(' ', (payload.get('description') or '')).strip()
    command = payload.get('command') or ''
    # Underscores and hyphens are word characters: "write_file" must read as "write file" or
    # the verb table misses it and the phrase falls back to the raw description.
    blob = re.sub(r'[_-]+', ' ', f'{desc} {command}').strip()
    target = ''
    kind = ''
    m = _SERVICE_RE.search(command) or _SERVICE_RE.search(desc)
    if m:
        target = f'the service {re.sub(r"\.(service|timer|socket)$", "", m.group(1))}'
        kind = 'service'
    if not target:
        m = _CONTAINER_RE.search(command) or _CONTAINER_RE.search(desc)
        if m and m.group(2):
            target = f'the container {m.group(2)}'
            kind = 'container'
        elif m:
            target = 'a container'
            kind = 'container'
    if not target:
        m = _PATH_RE.search(command) or _PATH_RE.search(desc)
        if m:
            target = _short_place(m.group(1))
            kind = 'file'
    if not target:
        m = _FILEISH_RE.search(command) or _FILEISH_RE.search(desc)
        if m:
            target = _short_place(m.group(1))
            kind = 'file'
    if kind == 'file' and ' in ' not in target:
        # The description often names the place in words rather than as a path.
        for hint, where in _PLACE_HINTS:
            if hint in desc.lower() or hint in command.lower():
                target = f'{target}{where}'
                break
    verb = _action_verb(blob)
    lead = _LEADING_VERB_RE.match(desc)
    if lead:
        # The description's own verb is the most trustworthy one ("Delete test2.txt; in your…").
        for regex, name in _VERB_RULES:
            if regex.search(lead.group(1)):
                verb = name
                break
    first_word = command.split()[0].split('/')[-1] if command.split() else ''
    # A plain file operation, a service or a container is named as such; anything else is a
    # command, and says so.
    file_action = bool(target) and (kind in ('service', 'container') or not command
                                    or first_word in FILE_COMMANDS
                                    or 'asked to write_file' in desc or 'asked to patch' in desc)
    if file_action and verb and verb != 'Run':
        return _STRAY_RE.sub(' ', f'{verb} {target}').strip()[:90]
    if command:
        purpose = _purpose(blob, target)
        if target and ' in ' in target and purpose.startswith(('uses git', 'changes files')):
            purpose = f'{purpose} ({target.split(" in ", 1)[0]})'
        return _STRAY_RE.sub(' ', f'Run a command that {purpose}').strip()[:90]
    if target:
        return _STRAY_RE.sub(' ', f'{verb or "Change"} {target}' if verb != 'Run' else target)[:90]
    cleaned = _STRAY_RE.sub(' ', desc).strip()
    return _scrub(cleaned or 'a flagged action')[:90]


def where_from_session(session_key: str) -> str:
    k = (session_key or '').lower()
    for name, label in (('telegram', 'your Telegram chat'), ('discord', 'your Discord chat'),
                        ('slack', 'your Slack chat'), ('signal', 'your Signal chat'),
                        ('whatsapp', 'your WhatsApp chat'), ('matrix', 'your Matrix room'),
                        ('cli', 'the command line')):
        if name in k:
            return label
    return 'your chat'


def human_age(seconds: float) -> str:
    minutes = max(1, int(round((seconds or 0) / 60.0)))
    if minutes < 60:
        return f'{minutes} minute' + ('s' if minutes != 1 else '')
    hours = minutes // 60
    return f'{hours} hour' + ('s' if hours != 1 else '')


def question_text(payload: dict, *, age_s: float = 0.0, where: str = '', inline: bool = False) -> str:
    """The spoken question. Inline means: asked inside a call that is already up.

    Short by design: the action and the target, then the two words that answer it. The long
    form read the whole path out loud and took thirteen seconds of a live call.
    """
    action = plain_action(payload)
    if inline:
        return f'{action}? Say approve or deny.'
    return (f'Sir, {human_age(age_s)} ago in {where or "your chat"}, an action was flagged for '
            f'approval: {action}. Say approve or deny.')


def reask_text() -> str:
    return 'Say approve or deny.'


def ack_text(decision: str) -> str:
    return ('Thank you, sir. That is approved.' if decision == 'yes'
            else 'Understood, sir. Refused.')


# --- the handshake -----------------------------------------------------------

def ensure_dirs() -> None:
    for d in (APPROVAL_DIR, ASK_DIR, RESULT_DIR, CANCEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(d, 0o700)
        except OSError:
            pass


# A request id becomes a file name, so it must never be a path. An id that does not look like an
# id is hashed rather than rejected: a legitimate but unusual id still works, and `../` cannot
# walk out of the approval folder.
_ID_SAFE = re.compile(r'^[A-Za-z0-9._-]{1,64}$')


def _safe_id(rid: str) -> str:
    rid = str(rid or '')
    if _ID_SAFE.match(rid) and rid not in ('.', '..'):
        return rid
    return 'x' + hashlib.sha256(rid.encode('utf-8', 'replace')).hexdigest()[:32]


def ask_path(rid: str) -> pathlib.Path:
    return ASK_DIR / f'{_safe_id(rid)}.json'


def result_path(rid: str) -> pathlib.Path:
    return RESULT_DIR / f'{_safe_id(rid)}.json'


def cancel_path(rid: str) -> pathlib.Path:
    return CANCEL_DIR / f'{_safe_id(rid)}.json'


def write_json(path: pathlib.Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=1))
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(path)


def read_json(path: pathlib.Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def write_ask(rid: str, payload: dict, *, age_s: float = 0.0, inline: bool = True) -> pathlib.Path:
    ensure_dirs()
    path = ask_path(rid)
    write_json(path, {'id': rid, 'payload': payload, 'age_s': age_s,
                      'text': question_text(payload, age_s=age_s, inline=inline),
                      'inline': inline, 'at': time.time()})
    return path


def take_ask() -> dict | None:
    """The oldest outstanding ask, consumed by the call that will ask it."""
    for path in sorted(ASK_DIR.glob('*.json'), key=lambda p: p.stat().st_mtime):
        data = read_json(path)
        try:
            path.unlink()
        except OSError:
            pass
        if data:
            return data
    return None


def write_result(rid: str, decision: str, transcripts: list, extra: dict | None = None) -> None:
    ensure_dirs()
    write_json(result_path(rid), {'id': rid, 'decision': decision,
                                  'transcripts': [_scrub(t) for t in (transcripts or [])],
                                  'at': time.time(), **(extra or {})})


def read_result(rid: str) -> dict | None:
    return read_json(result_path(rid))


def clear_result(rid: str) -> None:
    """Remove the handshake files for one request."""
    clear_handshake(rid)
    clear_cancel(rid)


def clear_handshake(rid: str) -> None:
    """Remove what the request *asked* and what it *answered*, but keep the cancelled
    marker: a call may still be asking, and it needs to learn the request is over."""
    for path in (result_path(rid), ask_path(rid)):
        try:
            path.unlink()
        except OSError:
            pass


def clear_cancel(rid: str) -> None:
    try:
        cancel_path(rid).unlink()
    except OSError:
        pass


def prune_stale(max_age_s: float = 3600) -> int:
    """Drop handshake files older than an hour so nothing accumulates."""
    removed = 0
    cutoff = time.time() - max_age_s
    for folder in (ASK_DIR, RESULT_DIR, CANCEL_DIR):
        try:
            for path in folder.glob('*.json*'):
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
        except OSError:
            pass
    return removed


def is_cancelled(rid: str) -> bool:
    return cancel_path(rid).exists()


def mark_cancelled(rid: str, why: str = 'answered elsewhere') -> None:
    ensure_dirs()
    if not cancel_path(rid).exists():
        write_json(cancel_path(rid), {'id': rid, 'why': why, 'at': time.time()})
