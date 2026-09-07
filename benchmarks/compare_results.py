#!/usr/bin/env python3
"""Compare two coding-suite JSON files without discarding per-scenario detail."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "ttft_s_median",
    "prefill_tok_s_approx_median",
    "decode_tok_s_median",
    "e2e_output_tok_s_median",
    "emitted_per_target_step_median",
    "power_w_mean",
    "power_w_peak",
    "vram_mib_peak",
)


def measured_runs(result: dict[str, Any], scenario: str) -> list[dict[str, Any]]:
    return [row for row in result["runs"] if row["scenario"] == scenario and not row["warmup"]]


def median_energy_j(result: dict[str, Any], scenario: str) -> float | None:
    values = []
    for row in measured_runs(result, scenario):
        power = row["gpu"].get("power_w_mean")
        if power is not None:
            values.append(float(power) * float(row["latency_s"]))
    return statistics.median(values) if values else None


def hashes(result: dict[str, Any], scenario: str, key: str) -> list[str]:
    return sorted({str(row[key]) for row in measured_runs(result, scenario)})


def compare(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_rows = {row["scenario"]: row for row in base["aggregates"]}
    candidate_rows = {row["scenario"]: row for row in candidate["aggregates"]}
    if set(base_rows) != set(candidate_rows):
        raise ValueError("result files contain different scenario sets")
    rows = []
    for scenario in base_rows:
        left, right = base_rows[scenario], candidate_rows[scenario]
        metrics = {}
        for metric in METRICS:
            a, b = left.get(metric), right.get(metric)
            metrics[metric] = {
                "base": a,
                "candidate": b,
                "delta_pct": ((b / a) - 1.0) * 100.0 if a not in (None, 0) and b is not None else None,
            }
        energy_a = median_energy_j(base, scenario)
        energy_b = median_energy_j(candidate, scenario)
        metrics["request_energy_j_median"] = {
            "base": energy_a,
            "candidate": energy_b,
            "delta_pct": ((energy_b / energy_a) - 1.0) * 100.0 if energy_a not in (None, 0) and energy_b is not None else None,
        }
        rows.append(
            {
                "scenario": scenario,
                "metrics": metrics,
                "prompt_hashes_identical": hashes(base, scenario, "prompt_sha256") == hashes(candidate, scenario, "prompt_sha256"),
                "output_hashes_identical": hashes(base, scenario, "output_sha256") == hashes(candidate, scenario, "output_sha256"),
                "base_cache_hit_tokens": left["cache_hit_tokens"],
                "candidate_cache_hit_tokens": right["cache_hit_tokens"],
            }
        )
    return {
        "base": base["meta"],
        "candidate": candidate["meta"],
        "scenarios": rows,
        "all_outputs_identical": all(row["output_hashes_identical"] for row in rows),
        "all_prompts_identical": all(row["prompt_hashes_identical"] for row in rows),
    }


def markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['base']['tag']} vs {result['candidate']['tag']}",
        "",
        "Positive percentages mean the candidate value is higher. Lower is better for TTFT and request energy; higher is better for throughput.",
        "",
        "| Scenario | TTFT Δ | Prefill Δ | Decode Δ | E2E Δ | Tok/step Δ | Energy Δ | Outputs equal |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in result["scenarios"]:
        metrics = row["metrics"]
        pct = lambda name: "—" if metrics[name]["delta_pct"] is None else f"{metrics[name]['delta_pct']:+.1f}%"
        lines.append(
            f"| {row['scenario']} | {pct('ttft_s_median')} | {pct('prefill_tok_s_approx_median')} | "
            f"{pct('decode_tok_s_median')} | {pct('e2e_output_tok_s_median')} | "
            f"{pct('emitted_per_target_step_median')} | {pct('request_energy_j_median')} | "
            f"{'yes' if row['output_hashes_identical'] else 'NO'} |"
        )
    lines += ["", f"All prompts identical: **{result['all_prompts_identical']}**. All outputs identical: **{result['all_outputs_identical']}**.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.base.read_text()), json.loads(args.candidate.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(".md").write_text(markdown(result))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
