#!/bin/bash
# Load the settings file and run call_me.py with a python that has baresipy installed.
#
#   ./call.sh "opening sentence"
#   ./call.sh --script spec.json          # a prepared script, as the scheduled calls use
#
# PHONE_ENV   path to the settings file   (default ./config.env)
# PHONE_PYTHON  the interpreter to use    (default python3)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

ENV_FILE="${PHONE_ENV:-./config.env}"
if [ ! -f "$ENV_FILE" ]; then
    echo "call.sh: no settings file at $ENV_FILE -- copy config.example.env to config.env and fill it in" >&2
    exit 2
fi

set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

exec "${PHONE_PYTHON:-python3}" call_me.py "$@"
