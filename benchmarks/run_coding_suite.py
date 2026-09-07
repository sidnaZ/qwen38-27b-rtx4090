#!/usr/bin/env python3
"""Reproducible, concurrency-one coding-assistant benchmark.

Prompts use checksum-verified corpus fixtures bundled with the repository.
Every measured request receives a distinct cache salt so prefix caching cannot
silently turn a cold-prefill arm into a cache-hit arm, while prompt tokens and
model output remain identical.
Results include raw per-request JSON, aggregates, and a compact Markdown report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
METRIC_NAMES = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:prompt_tokens_by_source_total",
    "vllm:generation_tokens_total",
)


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 1200) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def request_text(url: str, timeout: int = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode()


def run_text(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def build_context(scenario: dict[str, Any], corpus_dir: Path) -> str:
    target = int(scenario["context_chars"])
    if target == 0:
        return ""
    path = corpus_dir / scenario["context_file"]
    context = path.read_text()
    digest = hashlib.sha256(context.encode()).hexdigest()
    if len(context) != target or digest != scenario["context_sha256"]:
        raise ValueError(f"{scenario['id']}: corpus fixture length or SHA-256 mismatch: {path}")
    return context


def parse_metrics(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name_labels, _, raw = line.partition(" ")
        name = name_labels.split("{", 1)[0]
        if name not in METRIC_NAMES:
            continue
        labels = name_labels[len(name) :]
        key = name
        if 'source="' in labels:
            key += ":" + labels.split('source="', 1)[1].split('"', 1)[0]
        try:
            values[key] = values.get(key, 0.0) + float(raw)
        except ValueError:
            pass
    return values


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {key: after.get(key, 0.0) - before.get(key, 0.0) for key in set(before) | set(after)}


class GpuSampler:
    """Collect low-rate device telemetry without adding Python GPU dependencies."""

    def __init__(self, interval: float):
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, float | int | None]:
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval * 3))
        if not self.samples:
            return {"samples": 0, "power_w_mean": None, "power_w_peak": None, "vram_mib_peak": None, "gpu_util_mean": None, "gpu_util_peak": None, "temperature_c_peak": None}
        return {
            "samples": len(self.samples),
            "power_w_mean": statistics.fmean(x["power_w"] for x in self.samples),
            "power_w_peak": max(x["power_w"] for x in self.samples),
            "vram_mib_peak": max(x["vram_mib"] for x in self.samples),
            "gpu_util_mean": statistics.fmean(x["gpu_util"] for x in self.samples),
            "gpu_util_peak": max(x["gpu_util"] for x in self.samples),
            "temperature_c_peak": max(x["temperature_c"] for x in self.samples),
        }

    def _run(self) -> None:
        query = "power.draw,memory.used,utilization.gpu,temperature.gpu"
        while not self._stop.is_set():
            try:
                row = run_text("nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits").splitlines()[0]
                power, memory, util, temperature = (float(x.strip()) for x in row.split(","))
                self.samples.append({"power_w": power, "vram_mib": memory, "gpu_util": util, "temperature_c": temperature})
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self._stop.wait(self.interval)


def stream_chat(base_url: str, model: str, content: str, max_tokens: int, cache_salt: str, timeout: int) -> tuple[dict[str, Any], str]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 0,
        "cache_salt": cache_salt,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    usage: dict[str, Any] = {}
    finish_reason = None
    output: list[str] = []
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    first_token_at = first_token_at or time.perf_counter()
                    output.append(piece)
    ended = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("stream completed without a content token")
    return {
        "ttft_s": first_token_at - started,
        "latency_s": ended - started,
        "decode_window_s": ended - first_token_at,
        "usage": usage,
        "finish_reason": finish_reason,
    }, "".join(output)


def run_once(args: argparse.Namespace, scenario: dict[str, Any], context: str, run_index: int, warmup: bool) -> dict[str, Any]:
    cache_salt = hashlib.sha256(
        f"{args.cache_salt_prefix}:{scenario['id']}:{'w' if warmup else 'r'}:{run_index}".encode()
    ).hexdigest()
    content = ""
    if context:
        content += "Repository context follows:\n" + context + "\n\nEnd repository context.\n\n"
    content += scenario["instruction"]
    metrics_url = args.base_url.rsplit("/v1", 1)[0] + "/metrics"
    before = parse_metrics(request_text(metrics_url))
    sampler = GpuSampler(args.gpu_sample_interval)
    sampler.start()
    try:
        timing, output = stream_chat(args.base_url, args.model, content, int(scenario["max_tokens"]), cache_salt, args.timeout)
    finally:
        gpu = sampler.stop()
    after = parse_metrics(request_text(metrics_url))
    delta = metric_delta(before, after)
    usage = timing.pop("usage")
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    decode_window = float(timing["decode_window_s"])
    drafts = delta.get("vllm:spec_decode_num_drafts_total", 0.0)
    draft_tokens = delta.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    accepted = delta.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    return {
        "scenario": scenario["id"],
        "category": scenario["category"],
        "warmup": warmup,
        "run": run_index,
        "cache_salt_sha256": hashlib.sha256(cache_salt.encode()).hexdigest(),
        "prompt_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "output_chars": len(output),
        "finish_reason": timing["finish_reason"],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": timing["ttft_s"],
        "prefill_tok_s_approx": prompt_tokens / timing["ttft_s"] if timing["ttft_s"] else None,
        "latency_s": timing["latency_s"],
        "decode_tok_s": (completion_tokens - 1) / decode_window if completion_tokens > 1 and decode_window else None,
        "e2e_output_tok_s": completion_tokens / timing["latency_s"] if timing["latency_s"] else None,
        "spec": {
            "target_steps": drafts,
            "draft_tokens": draft_tokens,
            "accepted_tokens": accepted,
            "accepted_per_target_step": accepted / drafts if drafts else None,
            "emitted_per_target_step": 1.0 + accepted / drafts if drafts else None,
            "draft_acceptance": accepted / draft_tokens if draft_tokens else None,
        },
        "cache": {
            "local_compute_tokens": delta.get("vllm:prompt_tokens_by_source_total:local_compute", 0.0),
            "local_cache_hit_tokens": delta.get("vllm:prompt_tokens_by_source_total:local_cache_hit", 0.0),
        },
        "gpu": gpu,
    }


def median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.median(values) if values else None


def nested_values(rows: list[dict[str, Any]], section: str, key: str) -> list[float]:
    return [float(row[section][key]) for row in rows if row[section].get(key) is not None]


def aggregate(rows: list[dict[str, Any]], scenario: dict[str, Any]) -> dict[str, Any]:
    measured = [row for row in rows if row["scenario"] == scenario["id"] and not row["warmup"]]
    emitted = nested_values(measured, "spec", "emitted_per_target_step")
    power_mean = nested_values(measured, "gpu", "power_w_mean")
    power_peak = nested_values(measured, "gpu", "power_w_peak")
    vram_peak = nested_values(measured, "gpu", "vram_mib_peak")
    return {
        "scenario": scenario["id"],
        "category": scenario["category"],
        "runs": len(measured),
        "prompt_tokens_median": median(measured, "prompt_tokens"),
        "completion_tokens_median": median(measured, "completion_tokens"),
        "ttft_s_median": median(measured, "ttft_s"),
        "prefill_tok_s_approx_median": median(measured, "prefill_tok_s_approx"),
        "decode_tok_s_median": median(measured, "decode_tok_s"),
        "e2e_output_tok_s_median": median(measured, "e2e_output_tok_s"),
        "emitted_per_target_step_median": statistics.median(emitted) if emitted else None,
        "power_w_mean": statistics.fmean(power_mean) if power_mean else None,
        "power_w_peak": max(power_peak) if power_peak else None,
        "vram_mib_peak": max(vram_peak) if vram_peak else None,
        "cache_hit_tokens": sum(row["cache"]["local_cache_hit_tokens"] for row in measured),
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def markdown(meta: dict[str, Any], aggregates: list[dict[str, Any]]) -> str:
    lines = [
        f"# Coding benchmark: {meta['tag']}",
        "",
        f"- Git: `{meta['git_commit']}`",
        f"- Source corpus: `{meta['corpus_version']}`",
        f"- GPU: `{meta['gpu']}`",
        f"- Power limit: `{meta['power_limit_w']} W`",
        f"- Image: `{meta.get('image_id') or 'not supplied'}`",
        "",
        "| Scenario | Prompt tok | TTFT s | Approx prefill tok/s | Decode tok/s | E2E tok/s | Tok/step | Mean W | Peak VRAM MiB | Cache hits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        fmt = lambda key, digits=1: "—" if row[key] is None else f"{row[key]:.{digits}f}"
        lines.append(
            f"| {row['scenario']} | {fmt('prompt_tokens_median', 0)} | {fmt('ttft_s_median', 3)} | "
            f"{fmt('prefill_tok_s_approx_median', 0)} | {fmt('decode_tok_s_median')} | "
            f"{fmt('e2e_output_tok_s_median')} | {fmt('emitted_per_target_step_median', 2)} | "
            f"{fmt('power_w_mean')} | {fmt('vram_mib_peak', 0)} | {row['cache_hit_tokens']:.0f} |"
        )
    lines += ["", "TTFT includes request handling and first-token generation; approximate prefill throughput is prompt tokens divided by TTFT.", "Draft and verification time are not separately exported by this vLLM metrics endpoint.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="result name (required unless --list is used)")
    parser.add_argument(
        "--cache-salt-prefix",
        help="stable salt namespace; defaults to tag (set the same value across controlled A/B arms)",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:18020/v1")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--corpus-dir", type=Path, default=HERE)
    parser.add_argument("--workloads", type=Path, default=HERE / "coding_workloads.json")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results/coding-suite"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--scenarios", help="comma-separated scenario ids")
    parser.add_argument("--image-id")
    parser.add_argument("--config", action="append", default=[], metavar="KEY=VALUE", help="record an effective launch setting")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    args.cache_salt_prefix = args.cache_salt_prefix or args.tag

    scenarios = json.loads(args.workloads.read_text())
    if args.scenarios:
        wanted = set(args.scenarios.split(","))
        scenarios = [scenario for scenario in scenarios if scenario["id"] in wanted]
    if args.list:
        for scenario in scenarios:
            print(f"{scenario['id']}: {scenario['category']}")
        return 0
    if not args.tag:
        parser.error("--tag is required unless --list is used")
    if not scenarios or args.repeats < 1 or args.warmups < 0:
        parser.error("select at least one scenario, repeats >= 1, warmups >= 0")

    request_text(args.base_url.rsplit("/v1", 1)[0] + "/health")
    git_commit = run_text("git", "-C", str(REPO), "rev-parse", "HEAD")
    gpu = run_text("nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader")
    power_limit = float(run_text("nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits").splitlines()[0])
    config: dict[str, str] = {}
    for item in args.config:
        if "=" not in item:
            parser.error(f"--config must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        config[key] = value
    meta = {
        "tag": args.tag,
        "cache_salt_prefix": args.cache_salt_prefix,
        "created_unix": time.time(),
        "git_commit": git_commit,
        "working_tree_dirty": bool(run_text("git", "-C", str(REPO), "status", "--porcelain")),
        "corpus_version": "bundled-v1",
        "gpu": gpu,
        "power_limit_w": power_limit,
        "image_id": args.image_id,
        "config": config,
        "base_url": args.base_url,
        "model": args.model,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "server_version": request_json(args.base_url.rsplit("/v1", 1)[0] + "/version"),
        "server_models": request_json(args.base_url + "/models"),
        "limitations": ["TTFT is used as an approximate prefill latency", "vLLM does not export separate draft-model and verification durations"],
    }
    rows: list[dict[str, Any]] = []
    contexts = {scenario["id"]: build_context(scenario, args.corpus_dir) for scenario in scenarios}
    for scenario in scenarios:
        for index in range(args.warmups):
            row = run_once(args, scenario, contexts[scenario["id"]], index + 1, True)
            rows.append(row)
            print(f"WARM {scenario['id']} prompt={row['prompt_tokens']} ttft={row['ttft_s']:.3f}s decode={row['decode_tok_s'] or 0:.1f}", flush=True)
        for index in range(args.repeats):
            row = run_once(args, scenario, contexts[scenario["id"]], index + 1, False)
            rows.append(row)
            print(f"RUN  {scenario['id']} #{index + 1} prompt={row['prompt_tokens']} ttft={row['ttft_s']:.3f}s decode={row['decode_tok_s'] or 0:.1f} tok/step={row['spec']['emitted_per_target_step'] or 0:.2f}", flush=True)

    aggregates = [aggregate(rows, scenario) for scenario in scenarios]
    result = {"meta": meta, "workloads": scenarios, "runs": rows, "aggregates": aggregates}
    prefix = args.output_dir / args.tag
    atomic_json(prefix.with_suffix(".json"), result)
    prefix.with_suffix(".md").write_text(markdown(meta, aggregates))
    print(f"RESULT {prefix.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
