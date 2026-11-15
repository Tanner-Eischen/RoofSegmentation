#!/usr/bin/env bash
# Keep all artifacts on /workspace; stop compute after verified retrieval or timeout.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:?usage: supervise.sh ABSOLUTE_OUT MAX_WALL_SECONDS [bootstrap arguments]}"
WALL="${2:?a finite Pod wall-time limit is required}"
shift 2
[[ "$OUT" == /workspace/* ]] || { echo 'Output must be on persistent /workspace'; exit 64; }
[[ "$WALL" =~ ^[0-9]+$ && "$WALL" -ge 600 ]] || { echo 'Invalid wall-time limit'; exit 64; }
: "${RUNPOD_POD_ID:?Runpod Pod identity is required}"
CONTROL="$REPO_ROOT/runpod_control"
mkdir -p "$CONTROL"
exec 9>"$CONTROL/supervisor.lock"
flock -n 9 || { echo 'A supervisor is already active'; exit 73; }
cd "$REPO_ROOT"
if runpodctl pod stop --help >"$CONTROL/stop-help.txt" 2>&1; then
    STOP_STYLE=modern
elif runpodctl stop pod --help >"$CONTROL/stop-help.txt" 2>&1; then
    STOP_STYLE=legacy
else
    echo 'No verified Pod stop command; refusing to start billed training'
    exit 78
fi
stop_self() {
    if [[ "$STOP_STYLE" == modern ]]; then
        runpodctl pod stop "$RUNPOD_POD_ID"
    else
        runpodctl stop pod "$RUNPOD_POD_ID"
    fi
}
cleanup_after_ack() {
    # Removal is allowed only after the local watcher has verified every result.
    # If the Pod-scoped key cannot delete, retain its volume and stop compute.
    if [[ "${DELETE_AFTER_ACK:-0}" == 1 ]] && runpodctl pod delete --help >"$CONTROL/delete-help.txt" 2>&1; then
        if runpodctl pod delete "$RUNPOD_POD_ID"; then
            return 0
        fi
    fi
    if [[ "${DELETE_AFTER_ACK:-0}" == 1 ]] && runpodctl remove pod --help >"$CONTROL/delete-help.txt" 2>&1; then
        if runpodctl remove pod "$RUNPOD_POD_ID"; then
            return 0
        fi
    fi
    stop_self
}
export STOP_STYLE
if [[ "${1:-}" == --check-only ]]; then
    echo "Pod stop command verified ($STOP_STYLE); no watchdog, bootstrap, or stop action started"
    exit 0
fi
# A detached process remains armed even if the foreground job/session fails.
nohup bash -c '
    sleep "$1"
    date -u
    if [[ "$STOP_STYLE" == modern ]]; then
        exec runpodctl pod stop "$RUNPOD_POD_ID"
    else
        exec runpodctl stop pod "$RUNPOD_POD_ID"
    fi
' _ "$WALL" >"$CONTROL/walltime-stop.log" 2>&1 < /dev/null &
echo "$!" >"$CONTROL/watchdog.pid"
date -u >"$CONTROL/started_utc.txt"
set +e
bash scripts/runpod/bootstrap.sh --out "$OUT" "$@" >"$CONTROL/bootstrap.log" 2>&1
JOB_EXIT=$?
set -e
echo "$JOB_EXIT" >"$CONTROL/job_exit_code.txt"
ARCHIVE="$(dirname "$OUT")/$(basename "$OUT")_results.tar"
# Bootstrap failures can precede the scientific runner; retain their logs too.
if [[ ! -f "$ARCHIVE.sha256" ]]; then
    python3 - "$OUT" "$CONTROL/bootstrap.log" <<'PY'
import json, pathlib, runpy, shutil, sys
out=pathlib.Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
shutil.copyfile(sys.argv[2],out/'bootstrap_or_export_failure.log')
(out/'remote_status.json').write_text(json.dumps({'phase':'bootstrap_or_export_failed','full_training_started':None}))
helpers=runpy.run_path('scripts/runpod/remote_run.py',run_name='runpod_export_helper')
helpers['export_results'](out)
PY
fi
# The local watcher ACKs only after outer and per-file archive verification.
ACK="$OUT/retrieval_ack.json"
for ((attempt=0; attempt<90; attempt++)); do
    if [[ -f "$ACK" && -f "$ARCHIVE.sha256" ]] && python3 - "$ACK" "$ARCHIVE.sha256" <<'PY'
import json,pathlib,sys
ack=json.loads(pathlib.Path(sys.argv[1]).read_text())
expected=pathlib.Path(sys.argv[2]).read_text().split()[0]
raise SystemExit(0 if ack.get('verified') is True and ack.get('archive_sha256')==expected else 1)
PY
    then
        echo 'Verified local retrieval acknowledged; cleaning up temporary Pod'
        cleanup_after_ack
        exit "$JOB_EXIT"
    fi
    sleep 20
done
echo 'Retrieval window ended; stopping compute, retaining /workspace volume for recovery'
stop_self
exit "$JOB_EXIT"
