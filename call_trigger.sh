#!/usr/bin/env bash
# Place a call in the background, so whatever asked for it stays free.
#
#   ./call_trigger.sh "<what to say first>"
#   ./call_trigger.sh --script <spec.json>
#
# One call at a time. A clean exit says only that the driver was backgrounded -- never that the
# call is up; watch call_trigger.log and the driver for that.
#
#   DRY_RUN=1 ./call_trigger.sh "x"    # show the command without dialling
set -uo pipefail

if [ "$#" -eq 0 ]; then
    echo 'usage: call_trigger.sh "<opening sentence>" | call_trigger.sh --script <spec.json>'
    exit 2
fi

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
LOG="${PHONE_TRIGGER_LOG:-$HERE/call_trigger.log}"
cd "$HERE"

# One call at a time: the driver process owns the call, baresip is its SIP client.
# Match only a real driver process (`python call_me.py`); a bare 'call_me\.py' pattern also
# matches any wrapper shell whose own command line contains that string.
DRIVER=$(pgrep -f '[p]ython[0-9.]* .*[c]all_me\.py' 2>/dev/null | head -3 | tr '\n' ' ')
SIP=$(pgrep -x baresip 2>/dev/null | head -3 | tr '\n' ' ')
if [ -n "$DRIVER$SIP" ]; then
    echo "a call is already running (call_me.py: ${DRIVER:-none}; baresip: ${SIP:-none}) - not starting another"
    exit 3
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
    printf 'dry run: would run -- nohup ./call.sh %s >> %s 2>&1 &\n' "$*" "$LOG"
    exit 0
fi

: >> "$LOG" && chmod 600 "$LOG"
nohup ./call.sh "$@" >> "$LOG" 2>&1 &
echo "call starting (pid $!) - log: $LOG"
