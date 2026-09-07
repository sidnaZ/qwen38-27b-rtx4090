# Reproduction guide

Run every command from the repository root on native Linux with one RTX 4090.

## Storage override for an existing model

```bash
cp profiles/single-user.env profiles/single-user.local.env
```

Set `MODELS_DIR`, `CACHE_DIR`, `QUALITY_DATA_DIR`, and `BENCH_RESULTS_DIR` in the
local copy, then export:

```bash
export PROFILE_FILE=profiles/single-user.local.env
```

The profile must retain `PROFILE_NAME=single-user`. Use the same
`PROFILE_FILE` for preparation, launch, validation, and benchmarks.

## Build and prepare

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
docker version
docker compose version
scripts/prepare.sh
```

Preparation is resumable and idempotent. It builds the pinned container,
downloads the base checkpoint and release additions, quantizes the large heads
and MTP module, builds the draft vocabulary, and fetches the W4A16 DFlash2
drafter. Source and model pins are in `release-manifest.json`.

## Single-user verification

```bash
scripts/run-single-user.sh --set-power-limit
scripts/validate-single-user.sh single-user
scripts/benchmark-single-user.sh single-user reproduction
```

The validator checks the selected environment, installed patches, prepared
model, active DFlash2/Marlin/attention paths, fatal logs, inference, API 12/12,
power, and VRAM.

The C1 benchmark reads checksum-fixed fixtures from `benchmarks/corpus/`. Those
files contain the exact prompt bytes used for the published benchmark and do
not depend on inaccessible Git history. One warmup is discarded and three
cold-prefix repetitions are measured per scenario.

Quality evaluation is optional:

```bash
scripts/prepare-quality-data.sh
scripts/quality-single-user.sh single-user reproduction-quality
```

## Batch verification

Stop single-user before taking the GPU for batch:

```bash
scripts/stop-single-user.sh
scripts/run-batch.sh batch --set-power-limit
scripts/validate-batch.sh batch
```

Use `scripts/benchmark-batch.sh batch` to invoke the batch harness. Preserve the
workload parameters with every result: synchronized C32, distinct short prompts,
and 512 forced output tokens define the published quality-first row.

For the experimental profile:

```bash
scripts/stop-batch.sh batch
scripts/run-batch.sh batch-max --set-power-limit
scripts/validate-batch.sh batch-max
```

## Artifact policy

Generated files belong under the configured artifact directories and must not
be committed. Before committing, verify:

```bash
git status --short
git check-ignore -v .artifacts/models/example models/example \
  bench/results/example.json profiles/single-user.local.env
docker compose --env-file profiles/single-user.env -f compose.yml config --quiet
```

Stopping services retains prepared models and caches.
