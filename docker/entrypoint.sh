#!/bin/bash
# Container entrypoint. First argument selects what to run:
#   single   runtime/start-single-user.sh (DFlash2, low latency)
#   batch    runtime/start-batch.sh       (throughput)
#   prepare  docker/prepare.sh            (download + prepare /app/models)
#   verify   verify.sh [args]
#   <anything else> is exec'd as a command (e.g. bash)
# Before serving, docker/prepare.sh runs (idempotent: a state check and
# seconds when the model is already prepared, the download + requantization
# otherwise; PREPARE=0 skips it — this is what makes a bare `docker run` with
# an empty models volume work), then verify.sh --no-server, which aborts on
# FAIL (patches missing, ...); VERIFY=0 skips that.
set -e
cd /app
cmd=${1:-single}; shift || true
case "$cmd" in
  single|batch)
    if [ "${PREPARE:-1}" != "0" ]; then
      bash docker/prepare.sh
    fi
    if [ "${VERIFY:-1}" != "0" ]; then
      bash verify.sh --no-server || { echo "entrypoint: verify.sh FAILED — fix the above or set VERIFY=0"; exit 1; }
    fi
    if [ "$cmd" = single ]; then exec bash runtime/start-single-user.sh "$@"; else exec bash runtime/start-batch.sh "$@"; fi ;;
  prepare) exec bash docker/prepare.sh "$@" ;;
  verify)  exec bash verify.sh "$@" ;;
  *)       exec "$cmd" "$@" ;;
esac
