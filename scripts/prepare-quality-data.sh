#!/usr/bin/env bash
# Download the three public datasets consumed by quality_battery.py.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$SCRIPT_DIR/runtime-lib.sh"
load_profile single-user
require_host
"${COMPOSE[@]}" run --rm prepare bash -lc '
  set -e
  hf download Salesforce/wikitext --repo-type dataset \
    --revision b08601e04326c79dfdd32d625aee71d232d685c3 \
    --include "wikitext-2-raw-v1/test-*" --local-dir /app/bench/quality-data/wikitext
  hf download openai/gsm8k --repo-type dataset \
    --revision 740312add88f781978c0658806c59bc2815b9866 \
    --include "main/test-*" --local-dir /app/bench/quality-data/gsm8k
  hf download HuggingFaceFW/fineweb-2 --repo-type dataset \
    --revision af9c13333eb981300149d5ca60a8e9d659b276b9 \
    --include "data/dan_Latn/test/000_00000.parquet" --local-dir /app/bench/quality-data/fineweb2
'
echo "quality data is ready under $QUALITY_DATA_DIR"
