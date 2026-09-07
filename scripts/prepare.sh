#!/usr/bin/env bash
# Build the immutable local single-user image and materialize the pinned target/drafter models.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$SCRIPT_DIR/runtime-lib.sh"
load_profile single-user
require_host
"${COMPOSE[@]}" config --quiet
"${COMPOSE[@]}" build
"${COMPOSE[@]}" run --rm prepare
echo "release image and model artifacts are ready."
