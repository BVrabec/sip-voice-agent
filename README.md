# sip-voice-agent

An assistant that **phones you** and holds a real spoken conversation over SIP. It rings your
mobile, says something, listens to your answer, replies out loud — and lets you **interrupt it
mid-sentence**. No app on the phone and no telephony service in the middle: a small Python program
on a server, an ordinary SIP account, and a speech model.

```
your phone  ◀──SIP──▶  baresip  ◀──▶  call_me.py  ──▶  speech-to-text ──▶  your LLM  ──▶  text-to-speech
     ▲                                     │
     └────────────── spoken reply ─────────┘        (and you can cut in at any moment)
```

It was built for a personal assistant that calls its owner — a wake-up call, a "leave now or you
will be late" call, a briefing — but the engine knows nothing about why it is calling. Give it a
sentence and it holds a conversation.

## What makes it work rather than merely connect

- **Barge-in that is measured, not guessed.** Interrupting needs your voice above a threshold
  *for a number of seconds*, never elapsed time — a gate set too low fires on room noise and chops
  the reply in half.
- **It speaks while the model is still writing.** The reply is streamed, split into sentences, and
  each sentence is synthesised and played as it completes, so the first words arrive in about a
  second instead of after the whole answer.
- **The opening line is synthesised before the phone rings.** The first synthesis in a process is
  the cold one (1.4–1.7 s measured); doing it while the line is still ringing takes it off the
  critical path.
- **Speaking goes through baresip's `mixausrc` module**, which mixes a clip into the live encoding
  stream. The obvious route — switching the call's audio source at runtime — delivers only a
  fraction of a second of audio on baresip 4.x and then goes silent.
- **A spoken approval path.** If the assistant is about to do something risky, the telephone system
  itself asks you out loud, and only an unambiguous "approve" counts. Anything else, silence, or a
  hang-up is a no.
- **It will not ring you deaf.** Before dialling it checks that the speech account can actually
  transcribe; if it cannot, the call is not placed at all. A call that cannot hear you is worse
  than no call.
- **It hangs up on its own** when you stop talking, and it never leaves a line held open after a
  crash — every exit path writes an outcome.

Measured on the line this was built for, per turn, to the first sound:

| stage | before | after the streaming and turn-end work |
|---|---|---|
| median turn | 4.34 s | **≈2.2 s** |
| of which the LLM | 2.22 s (51%) | — |
| first sentence, with the model's reasoning left on | 2.04 s median / 11.2 s worst | **0.74 s / 1.87 s** with it off |

The audio path is the last thing to touch. Most of the wait is the model, then transcription, then
the silence it waits for before deciding you have finished.

## What is in here

| file | what it is |
|---|---|
| `call_me.py` | the whole engine: register, dial, listen, transcribe, answer, speak, hang up |
| `approval_voice.py` | how a gated action is worded and how your spoken answer is judged |
| `call.sh` | loads the settings file and runs the engine |
| `call_trigger.sh` | places a call in the background, one at a time, never two |
| `config.example.env` | every setting, with the two you must fill in at the top |
| `requirements.txt` | `baresipy` and `requests` |

**Not in here, on purpose:** the scheduling layer. In the deployment this came from, separate
private code decides *when* to call (a wake-up, a departure time, a weekend check-in) and writes a
small JSON spec. The engine reads that spec with `--script` — see *Scheduled calls* below. That
split is the point: the engine is generic, the reasons to call are personal.

## Requirements

- A Linux server with no sound card needed — the audio goes through files, not devices.
- A **SIP account** and the number you want it to ring (any provider: Linphone, a PBX, an Asterisk
  of your own, a carrier's SIP trunk).
- **baresip 4.x built from source** (see below — the distribution package will not register).
- **`ffmpeg`** on the PATH: every speech clip is converted and level-matched with it.
- Python 3.9+ with `baresipy` and `requests`.
- An **OpenRouter** key for speech-to-text and text-to-speech.
- Any **OpenAI-compatible chat endpoint** for the replies — a Hermes agent, OpenAI, OpenRouter,
  Ollama, anything.

## Install

### 1. baresip, from a release tag — not from your distribution

Debian and Ubuntu ship baresip 1.1.0, whose bundled libre rejects every digest algorithm but MD5,
while most providers offer SHA-256 first. Registration then dies with
`Register: Function not implemented`, and no package upgrade fixes it.

```bash
apt install -y cmake pkg-config build-essential libssl-dev libsndfile1-dev libopus-dev zlib1g-dev ffmpeg
git ls-remote --tags --refs https://github.com/baresip/re       # pick the latest release tag
git ls-remote --tags --refs https://github.com/baresip/baresip
```

Build `re` first, then `baresip`, each with
`cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc) && cmake --install build`,
then `ldconfig`. (`rem` is not needed for baresip 4.x — it was folded into libre.)

Verify with `which -a baresip` and `timeout 5 baresip -v`. **Never run `baresip -v` without the
timeout** — it prints the version and then sits in its interactive menu.

### 2. Let baresip write its own config, then edit it

baresipy ships a template aimed at old baresip and, if `~/.baresipy/config` is missing, writes it —
pointing `module_path` at the distribution modules. A modern binary then fails to load every module
with `undefined symbol`, and **no registration is attempted at all**. Generate the native config
first:

```bash
mv ~/.baresipy ~/.baresipy.old 2>/dev/null; mkdir -p ~/.baresipy
timeout 8 baresip -f ~/.baresipy -t 3 </dev/null
```

Then edit `~/.baresipy/config`:

| line | value | why |
|---|---|---|
| `module aufile.so` | uncomment | the headless audio source and player |
| `module sndfile.so` | uncomment | needed for the receive/transmit dumps |
| `module opus.so` | uncomment | most softphones negotiate opus |
| `module alsa.so` | comment out | there is no sound card on a server |
| `module mixausrc.so` | add to the audio-filter list | **this is how speech is played** |
| `audio_source` | `aufile,<path>/silence.wav` | a long, stream-format-matched silence file (see below) |
| `audio_player`, `audio_alert` | `aufile,/dev/null` | otherwise playback fails and the answered call drops |
| `snd_path` | uncomment | so the dumps land where baresipy looks for them |

The silence file matters more than it looks. It must be **long** (900 s, not 60) and match the
stream format of the call — 48 kHz stereo, 16-bit. A 60-second file runs out mid-call, and a file
in a different format changes the stream mid-call and kills the encoder. One way to make it:

```bash
ffmpeg -f lavfi -i anullsrc=r=48000:cl=stereo -t 900 -c:a pcm_s16le ~/.baresipy/silence.wav
```

### 3. The Python side

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

### 4. Settings

```bash
cp config.example.env config.env
chmod 600 config.env        # it holds the SIP password and two API keys
$EDITOR config.env          # SIP_USER, SIP_PASS, TARGET_SIP, OPENROUTER_API_KEY, HERMES_*
```

### 5. Make it call you

```bash
./call.sh "Hello, this is your assistant calling."     # dials, speaks, listens, answers
```

Watch it with the log the trigger writes:

```bash
./call_trigger.sh "Hello, this is your assistant calling."   # background, one at a time
tail -f call_trigger.log
```

## Scheduled calls — the `--script` interface

Anything that decides *when* to call can hand the engine a prepared script. The JSON spec:

```json
{
  "lines": ["Good morning. It is six o'clock.", "You have three things today."],
  "question": "Shall I read them out?",
  "reminder": "Do not forget the form.",
  "context": "anything the assistant should know for this call",
  "closing_idle_s": 8,
  "tts_slow": 1.2,
  "listen_silence_s": 2.0,
  "max_s": 600,
  "result": "/path/to/outcome.json"
}
```

```bash
./call_trigger.sh --script /path/to/spec.json
```

With a spec the lines are synthesised while the phone is ringing and played on answer, the reminder
is spoken on every exit path the call still controls, and the outcome
(`answered` / `answered_silent` / `not_answered`) is written for whatever asked for the call.
`tts_slow` is a per-call pace (the morning briefing reads slower); `listen_silence_s` is a per-call
turn-end silence — raise it for a call that expects a long answer, keep it at half a second for one
that expects yes or no. Without a spec, nothing in this path changes: an ordinary call is an
ordinary call.

## Using it with Hermes

The replies can come from a [Hermes](https://hermes-agent.nousresearch.com/docs) agent, which is
what this was built against — then the voice on the phone is the same assistant you chat with, with
its memory, its skills and its tools.

1. On the Hermes host, make sure the **API server** platform is enabled (the dashboard lists it as
   `API_server`). It listens on `127.0.0.1:8642` by default.
2. Mint an API key for the caller.
3. In `config.env`:

   ```env
   HERMES_API_URL=http://127.0.0.1:8642/v1/chat/completions
   HERMES_API_KEY=<the key>
   HERMES_MODEL=hermes-agent
   ```

4. Restart nothing — `call.sh` sources `config.env` into the environment and the engine reads it at
   start. Place a call and the agent answers it.

Two notes from doing this for real:

- **Turn reasoning off for the line** (`PHONE_REASONING=none`, and check that it was actually
  honoured). Nobody hears reasoning tokens; they are pure delay, and they were the single largest
  latency lever measured here — 40 of 40 requests at zero reasoning tokens, the first sentence from
  2.04 s to 0.74 s.
- **A call must be recognisable to the agent.** If the agent has a rule that asks before risky
  actions, a call has to be identified as a call, or the gate treats it as an unattended API client
  and refuses everything. `call_me.py` sends `X-Hermes-Call: 1` and writes a live-call marker file
  that `approval_voice.py` reads, which is what lets the same gate ask *out loud* instead.

### A prompt you can hand to an agent

> Install this repository for me on this machine: clone it, build baresip 4.x from a release tag as
> the README describes, generate `~/.baresipy/config` with `baresip -f ~/.baresipy -t 3` and apply
> the config edits in the README's table, make a 900-second 48 kHz stereo silence file at
> `~/.baresipy/silence.wav`, create a venv and install `requirements.txt`. Then stop and ask me for
> the SIP account, the number to call and the API keys, write `config.env` from
> `config.example.env` with mode 600, and place one test call with `./call.sh "Hello."`. Report what
> the log said, not what you expected it to say.

## Security notes

- The settings file holds a SIP password and two API keys. Keep it at mode 600; `call.sh` reads it
  into the environment and never prints it, and the engine never writes a credential to a log.
- baresip's own console echoes the account line **including the password** character by character,
  and baresipy logs every raw line at INFO. `call_me.py` installs a filter that masks `auth_pass=`
  before anything else runs — keep it there if you edit the file.
- The engine makes outbound connections only: to the SIP provider, to your speech provider, and to
  your LLM. Nothing needs to reach it, and no port is opened on the machine.
- Give the speech provider its own key with a spend limit. A call is cheap but a broken loop is not.
- A request id becomes a **file name** inside the approval folder, so an id that does not look like
  one is hashed rather than used as given: `../` in an id cannot walk out of that folder, and the
  folder itself is created `0700`.

## Troubleshooting the things that actually go wrong

- **It rings and drops the moment you answer** → the audio source. Either `audio_source` is not a
  long, format-matched `aufile` (see the table above), or baresip fell back to `alsa,default`.
- **It answers and then carries silence** → the speech command reached baresip's menu parser
  instead of its command parser. Every command must carry a **leading `/`**; without it
  `mixausrc_enc_start …` is read as the mute command and the line goes quiet while everything else
  in the log looks healthy.
- **`488 Not acceptable here`** → the far end wants SRTP. Set `SIP_MEDIAENC=srtp-mand` and enable
  `module srtp.so`.
- **`403` / `Register: Function not implemented`** → the wrong baresip (see step 1) or a wrong
  password.
- **`408 Request Timeout` after about 45 seconds** → nobody picked up. That is the carrier's
  no-answer timer, not a fault in your setup.
- **The call ends with `Connection reset by peer`** → normally the human hanging up. Only chase it
  if a call ever ends with nobody on the line.
- **It talks over you, or stops when you breathe** → the barge-in gate and the turn-end silence are
  the two knobs. Both are in `config.env`.
- **A call that hangs up by itself after an audio error** → baresipy's own error handling calls
  `hang()` when it sees an audio-source failure. `call_me.py` subclasses that to log only; an audio
  error raised by our own source switching must never end a call.

## Licence

MIT — see `LICENSE`.
