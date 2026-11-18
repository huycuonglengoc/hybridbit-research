#!/usr/bin/env python3
"""Run CPU-only, GPU-only, static-hybrid, or dynamic-hybrid benchmarks.

Designed for a Google Colab runtime with PyTorch and an optional NVIDIA GPU.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

from hetero_pipeline.metrics import cuda_preflight, environment_report, export_samples, write_csv, write_json
from hetero_pipeline.runtime import BenchmarkConfig, BenchmarkRunner, TrialResult
from hetero_pipeline.workloads import build_workload


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def fraction(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", "cpu", "gpu", "hybrid-static", "hybrid-dynamic"], default="all")
    parser.add_argument("--workload", choices=["elementwise", "reduction", "matmul"], default="elementwise")
    parser.add_argument("--tasks", type=positive_int, default=24, help="number of deterministic task batches per trial")
    parser.add_argument("--batch-size", type=positive_int, default=8192, help="rows per task for elementwise/reduction")
    parser.add_argument("--width", type=positive_int, default=1024, help="columns for elementwise/reduction")
    parser.add_argument("--matrix-dim", type=positive_int, default=1024, help="square matrix edge for matmul")
    parser.add_argument("--trials", type=positive_int, default=3)
    parser.add_argument("--warmup", type=int, default=1, help="unreported trials before each measured mode")
    parser.add_argument("--cpu-workers", type=positive_int, default=min(2, os.cpu_count() or 1))
    parser.add_argument("--cpu-threads", type=positive_int, default=1, help="PyTorch intra-op CPU threads")
    parser.add_argument("--gpu-inflight", type=positive_int, default=2, help="bounded GPU batches/buffers in flight")
    parser.add_argument("--static-cpu-share", type=fraction, default=0.25)
    parser.add_argument("--min-gpu-batch", type=positive_int, default=2048)
    parser.add_argument("--hysteresis-percent", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dependency-chain", action="store_true", help="make each task depend on the prior task")
    parser.add_argument("--validate-tasks", type=int, default=3, help="full CPU oracle checks per trial; 0 disables")
    parser.add_argument("--no-pin-memory", action="store_true", help="disable pinned host buffers for GPU transfers")
    parser.add_argument("--cuda-device", type=int, default=0, help="CUDA device index (Colab normally exposes 0)")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail instead of running a CPU-only subset when CUDA preflight does not pass",
    )
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="allow TensorFloat-32 matrix math on supported Ampere-or-newer GPUs",
    )
    parser.add_argument("--no-gpu-fallback", action="store_true", help="fail hybrid work instead of retrying a failed GPU task on CPU")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser


def print_summary(summary: dict[str, object]) -> None:
    latency = summary["latency_ms"]  # type: ignore[index]
    print(
        f"  {summary['mode']:<15} trial {summary['trial_index']}: "
        f"{summary['throughput_tasks_per_s']:.2f} tasks/s | "  # type: ignore[index]
        f"median {latency['median']:.2f} ms | p95 {latency['p95']:.2f} ms | "  # type: ignore[index]
        f"validation {'PASS' if summary['validation_passed'] else 'FAIL'}"  # type: ignore[index]
    )


def aggregate(summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    by_mode: dict[str, list[dict[str, object]]] = {}
    for summary in summaries:
        by_mode.setdefault(str(summary["mode"]), []).append(summary)
    rows: list[dict[str, object]] = []
    for mode, values in by_mode.items():
        throughputs = [float(value["throughput_tasks_per_s"]) for value in values]
        rows.append(
            {
                "mode": mode,
                "trials": len(values),
                "throughput_tasks_per_s_median": statistics.median(throughputs),
                "throughput_tasks_per_s_stdev": statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0,
                "all_validation_passed": all(bool(value["validation_passed"]) for value in values),
            }
        )
    return rows


def main() -> int:
    args = build_parser().parse_args()
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.cuda_device < 0:
        raise SystemExit("--cuda-device must be non-negative")
    workload = build_workload(args.workload, args.width, args.matrix_dim)
    config = BenchmarkConfig(
        workload=workload,
        task_count=args.tasks,
        batch_size=args.batch_size,
        cpu_workers=args.cpu_workers,
        cpu_threads=args.cpu_threads,
        gpu_inflight=args.gpu_inflight,
        static_cpu_share=args.static_cpu_share,
        min_gpu_batch=args.min_gpu_batch,
        hysteresis_percent=args.hysteresis_percent,
        seed=args.seed,
        dependency_chain=args.dependency_chain,
        validate_tasks=args.validate_tasks,
        pin_memory=not args.no_pin_memory,
        gpu_fallback=not args.no_gpu_fallback,
        cuda_device=args.cuda_device,
        allow_tf32=args.allow_tf32,
    )
    runner = BenchmarkRunner(config)
    requested = [args.mode] if args.mode != "all" else ["cpu", "gpu", "hybrid-static", "hybrid-dynamic"]
    gpu_requested = any(mode != "cpu" for mode in requested)
    cuda_status = cuda_preflight(args.cuda_device) if gpu_requested or args.require_cuda else {"ok": runner.has_gpu()}
    if gpu_requested:
        if cuda_status["ok"]:
            device = cuda_status.get("device", {})
            print(f"CUDA preflight passed: {device.get('name', f'cuda:{args.cuda_device}')}")
        else:
            print("CUDA preflight failed: " + "; ".join(cuda_status.get("errors", [])), file=sys.stderr)
    if (args.require_cuda or args.mode != "all") and not cuda_status["ok"]:
        return 1
    # A visible CUDA device is not enough; only use the GPU modes after a real
    # asynchronous copy-and-kernel test completed successfully.
    modes = [mode for mode in requested if mode == "cpu" or cuda_status["ok"]]
    skipped = sorted(set(requested) - set(modes))
    if skipped:
        print("CUDA unavailable; skipped: " + ", ".join(skipped), file=sys.stderr)

    run_dir = args.output_dir / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "environment.json", environment_report(args.cuda_device))
    write_json(run_dir / "cuda_preflight.json", cuda_status)
    write_json(run_dir / "config.json", {key: value for key, value in vars(args).items() if key != "output_dir"})

    trial_results: list[TrialResult] = []
    for mode in modes:
        for warmup_index in range(args.warmup):
            print(f"Warming up {mode} ({warmup_index + 1}/{args.warmup})...", flush=True)
            runner.run(mode, trial_index=0)  # type: ignore[arg-type]
        for trial_index in range(1, args.trials + 1):
            print(f"Running {mode} ({trial_index}/{args.trials})...", flush=True)
            trial = runner.run(mode, trial_index=trial_index)  # type: ignore[arg-type]
            trial_results.append(trial)
            summary = trial.summary()
            print_summary(summary)
            write_csv(run_dir / f"trace_{mode}_trial_{trial_index}.csv", [item.trace_row(trial.started_at) for item in trial.tasks])
            export_samples(run_dir / f"resources_{mode}_trial_{trial_index}.csv", trial.resource_samples)

    summaries = [trial.summary() for trial in trial_results]
    report = {
        "environment": environment_report(args.cuda_device),
        "cuda_preflight": cuda_status,
        "config": {key: value for key, value in vars(args).items() if key != "output_dir"},
        "trials": summaries,
        "aggregate": aggregate(summaries),
    }
    write_json(run_dir / "summary.json", report)
    write_csv(run_dir / "summary.csv", report["aggregate"])
    print(f"\nResults written to: {run_dir.resolve()}")
    if any(not summary["validation_passed"] for summary in summaries):
        print("One or more correctness checks failed; inspect summary.json and trace files.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
