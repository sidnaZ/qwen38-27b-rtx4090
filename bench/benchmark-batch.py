#!/usr/bin/env python3
"""Controlled batch concurrency sweep against an OpenAI-compatible vLLM server.

This harness has no vLLM client dependency, so it runs from the host against
the containerized server.  It records request latency, TTFT, steady-state
decode counters, scheduler/KV gauges, and GPU memory/power/utilization.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path


COUNTERS = (
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_total",
    "vllm:iteration_tokens_total_sum",
    "vllm:iteration_tokens_total_count",
    "vllm:num_preemptions_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_drafts_total",
)
GAUGES = (
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
)
FILLER = (
    "The RTX 4090 has 24 GB of GDDR6X memory and a high-throughput Ada GPU. "
    "Batch decoding changes matrix shapes, scheduler pressure, and cache use. "
)
TOKENS_PER_FILLER = 41  # approximate; authoritative counts come from API usage


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


class Sweep:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.api = f"http://{args.host}:{args.port}"
        self.headers = {"Content-Type": "application/json"}
        if args.api_key:
            self.headers["Authorization"] = f"Bearer {args.api_key}"

    def get(self, path: str, timeout: float = 20) -> str:
        req = urllib.request.Request(self.api + path, headers=self.headers)
        return urllib.request.urlopen(req, timeout=timeout).read().decode()

    def metrics(self) -> dict[str, float]:
        text = self.get("/metrics")
        values: dict[str, float] = {}
        for name in COUNTERS + GAUGES:
            matches = re.findall(
                rf"^{re.escape(name)}(?:\{{[^}}]*\}})? ([0-9.eE+-]+)$", text, re.M
            )
            values[name.removeprefix("vllm:")] = sum(map(float, matches)) if matches else 0.0
        return values

    def prompt(self, request_id: int, salt: str) -> str:
        count = max(1, round(self.args.ctx / TOKENS_PER_FILLER))
        prefix = f"Request {request_id}; independent salt {salt}. "
        return (
            prefix
            + FILLER * count
            + "\nExplain the throughput implications carefully and finish with a summary."
        )

    def stream(self, request_id: int, salt: str, results: dict[int, dict], first: dict[int, float]) -> None:
        body = json.dumps(
            {
                "model": self.args.model,
                "messages": [{"role": "user", "content": self.prompt(request_id, salt)}],
                "max_tokens": self.args.out,
                "temperature": 0.0,
                "ignore_eos": True,
                "seed": self.args.seed + request_id,
                "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode()
        req = urllib.request.Request(
            self.api + "/v1/chat/completions", data=body, headers=self.headers
        )
        started = time.perf_counter()
        first_at = None
        last_at = None
        usage: dict = {}
        error = None
        try:
            with urllib.request.urlopen(req, timeout=self.args.timeout) as response:
                for raw in response:
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    event = json.loads(line[6:])
                    if event.get("usage"):
                        usage = event["usage"]
                    choices = event.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        if delta.get("content") or delta.get("reasoning_content"):
                            now = time.perf_counter()
                            if first_at is None:
                                first_at = now
                                first[request_id] = now
                            last_at = now
        except Exception as exc:  # preserve all successful peers and report the arm failed
            error = f"{type(exc).__name__}: {exc}"
        ended = time.perf_counter()
        completion = int(usage.get("completion_tokens") or 0)
        results[request_id] = {
            "request_id": request_id,
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": completion,
            "latency_s": ended - started,
            "ttft_s": first_at - started if first_at else None,
            "decode_s": last_at - first_at if first_at and last_at else None,
            "decode_tok_s": (
                (completion - 1) / (last_at - first_at)
                if completion > 1 and first_at and last_at and last_at > first_at
                else None
            ),
            "error": error,
        }

    @staticmethod
    def gpu_sample() -> dict[str, float] | None:
        command = [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        try:
            fields = subprocess.check_output(command, text=True, timeout=5).strip().split(",")
            return {
                "memory_mib": float(fields[0]),
                "util_pct": float(fields[1]),
                "power_w": float(fields[2]),
            }
        except Exception:
            return None

    def sample(self, stop: threading.Event, peaks: dict[str, float], samples: list[dict]) -> None:
        while not stop.is_set():
            try:
                metric = self.metrics()
                peaks["kv_pct"] = max(peaks["kv_pct"], 100 * metric["kv_cache_usage_perc"])
                peaks["active"] = max(peaks["active"], metric["num_requests_running"])
                peaks["waiting"] = max(peaks["waiting"], metric["num_requests_waiting"])
            except Exception:
                pass
            gpu = self.gpu_sample()
            if gpu:
                samples.append(gpu)
                peaks["memory_mib"] = max(peaks["memory_mib"], gpu["memory_mib"])
                peaks["util_pct"] = max(peaks["util_pct"], gpu["util_pct"])
                peaks["power_w"] = max(peaks["power_w"], gpu["power_w"])
            stop.wait(self.args.sample_interval)

    def run_once(self, concurrency: int, repetition: int) -> dict:
        results: dict[int, dict] = {}
        first: dict[int, float] = {}
        peaks = {k: 0.0 for k in ("kv_pct", "active", "waiting", "memory_mib", "util_pct", "power_w")}
        samples: list[dict] = []
        stop = threading.Event()
        monitor = threading.Thread(target=self.sample, args=(stop, peaks, samples), daemon=True)
        before = self.metrics()
        monitor.start()
        salt = f"{self.args.label}-c{concurrency}-r{repetition}-{int(time.time())}"
        workers = [
            threading.Thread(target=self.stream, args=(i, salt, results, first), daemon=True)
            for i in range(concurrency)
        ]
        started = time.perf_counter()
        for worker in workers:
            worker.start()

        decode_before = None
        decode_started = None
        decode_after = None
        decode_ended = None
        while any(worker.is_alive() for worker in workers):
            if decode_before is None and len(first) == concurrency:
                decode_before = self.metrics()
                decode_started = time.perf_counter()
            if decode_after is None and len(results) >= 1:
                decode_after = self.metrics()
                decode_ended = time.perf_counter()
            time.sleep(0.02)
        for worker in workers:
            worker.join()
        ended = time.perf_counter()
        after = self.metrics()
        stop.set()
        monitor.join(timeout=2)

        successful = [r for r in results.values() if not r["error"] and r["completion_tokens"]]
        full_window = bool(
            decode_before
            and decode_after
            and decode_started is not None
            and decode_ended is not None
            and decode_ended > decode_started
        )
        if not full_window:
            decode_before, decode_started = before, started
            decode_after, decode_ended = after, ended
        window_s = decode_ended - decode_started
        diff = lambda key: decode_after[key] - decode_before[key]
        all_diff = lambda key: after[key] - before[key]
        latencies = [1000 * r["latency_s"] for r in successful]
        ttfts = [1000 * r["ttft_s"] for r in successful if r["ttft_s"] is not None]
        request_rates = [r["decode_tok_s"] for r in successful if r["decode_tok_s"] is not None]
        completion_tokens = sum(r["completion_tokens"] for r in successful)
        passes = diff("iteration_tokens_total_count")
        gpu_power = [sample["power_w"] for sample in samples]
        gpu_util = [sample["util_pct"] for sample in samples]
        return {
            "concurrency": concurrency,
            "repetition": repetition,
            "successful": len(successful),
            "failures": concurrency - len(successful),
            "errors": [r["error"] for r in results.values() if r["error"]],
            "wall_s": ended - started,
            "completion_tokens": completion_tokens,
            "aggregate_e2e_tok_s": completion_tokens / (ended - started),
            "aggregate_decode_tok_s": diff("generation_tokens_total") / window_s,
            "full_decode_window": full_window,
            "per_request_decode_tok_s_mean": statistics.fmean(request_rates) if request_rates else None,
            "per_request_decode_tok_s_median": statistics.median(request_rates) if request_rates else None,
            "latency_ms_median": statistics.median(latencies) if latencies else None,
            "latency_ms_p95": percentile(latencies, 0.95),
            "ttft_ms_median": statistics.median(ttfts) if ttfts else None,
            "ttft_ms_p95": percentile(ttfts, 0.95),
            "tokens_per_pass": diff("iteration_tokens_total_sum") / passes if passes else None,
            "ms_per_pass": 1000 * window_s / passes if passes else None,
            "preemptions": all_diff("num_preemptions_total"),
            "accepted_tokens_per_step": (
                1 + all_diff("spec_decode_num_accepted_tokens_total") / all_diff("spec_decode_num_drafts_total")
                if all_diff("spec_decode_num_drafts_total")
                else 1.0
            ),
            "peak_active_sequences": peaks["active"],
            "peak_waiting_sequences": peaks["waiting"],
            "peak_kv_pct": peaks["kv_pct"],
            "peak_vram_mib": peaks["memory_mib"],
            "gpu_util_pct_mean": statistics.fmean(gpu_util) if gpu_util else None,
            "gpu_util_pct_peak": peaks["util_pct"],
            "power_w_mean": statistics.fmean(gpu_power) if gpu_power else None,
            "power_w_peak": peaks["power_w"],
            "tok_per_joule": (
                completion_tokens / ((ended - started) * statistics.fmean(gpu_power))
                if gpu_power and statistics.fmean(gpu_power) > 0
                else None
            ),
            "requests": sorted(results.values(), key=lambda item: item["request_id"]),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "18020")))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--n", default="1,2,4,8,12,16,32,64")
    parser.add_argument("--ctx", type=int, default=128)
    parser.add_argument("--out", type=int, default=512)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=260906)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--label", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    sweep = Sweep(args)
    sweep.get("/health")
    rows = []
    for concurrency in [int(value) for value in args.n.split(",")]:
        for repetition in range(1, args.reps + 1):
            row = sweep.run_once(concurrency, repetition)
            rows.append(row)
            print(
                f"{args.label} C{concurrency} r{repetition}: "
                f"decode={row['aggregate_decode_tok_s']:.1f} e2e={row['aggregate_e2e_tok_s']:.1f} "
                f"latency p50/p95={row['latency_ms_median']:.0f}/{row['latency_ms_p95']:.0f} ms "
                f"TTFT p50/p95={row['ttft_ms_median']:.0f}/{row['ttft_ms_p95']:.0f} ms "
                f"VRAM={row['peak_vram_mib']:.0f} MiB power={row['power_w_mean']:.0f} W "
                f"fail={row['failures']}",
                flush=True,
            )
    payload = {
        "schema": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "label": args.label,
        "host": args.host,
        "port": args.port,
        "model": args.model,
        "context_tokens_requested": args.ctx,
        "output_tokens_requested": args.out,
        "rows": rows,
    }
    target = Path(args.json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
