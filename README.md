# Heterogeneous CPU/GPU pipeline benchmark

This repository implements the attached plan as a compact PyTorch framework
for Google Colab. It benchmarks deterministic, generic numeric work in four
modes:

- `cpu`: bounded CPU worker pool.
- `gpu`: asynchronous CUDA submissions with explicit H2D, compute, and D2H
  timings.
- `hybrid-static`: a fixed, interleaved CPU/GPU split.
- `hybrid-dynamic`: a measured-cost scheduler with queue estimates, a minimum
  GPU batch size, and hysteresis.

It is intentionally not a claim that hybrid execution is always faster. The
reported throughput must beat the best single-device baseline on the same
inputs to be useful.

## Run on Google Colab

1. In Colab, choose **Runtime → Change runtime type → T4 GPU** (or another
   GPU). Verify it with `!nvidia-smi`.
2. Upload this folder, clone its repository, or use the Files sidebar to place
   it under `/content/sep3`.
3. In a Colab cell, run:

```bash
%cd /content/sep3
!pip -q install -r requirements.txt
!python3 run_benchmark.py --mode all --require-cuda --workload elementwise --tasks 24 --batch-size 8192 --width 1024 --trials 3 --warmup 1
```

For a quick smoke test:

```bash
!python3 run_benchmark.py --mode all --tasks 6 --batch-size 1024 --width 256 --trials 1 --warmup 0
```

For GPU-heavy matrix multiplication (reduce `--matrix-dim` if the Colab GPU is
small):

```bash
!python3 run_benchmark.py --mode all --workload matmul --tasks 12 --matrix-dim 1024 --trials 3 --warmup 1 --static-cpu-share 0.20
```

For a dependency-chain/DAG overhead check:

```bash
!python3 run_benchmark.py --mode hybrid-dynamic --dependency-chain --tasks 12 --batch-size 4096 --width 512
```

The script skips GPU modes safely if CUDA is unavailable, so the CPU baseline
also works on a non-GPU runtime.

`--require-cuda` is recommended for a Colab CPU/GPU comparison: it terminates
with an explanation if the runtime has no attached GPU, PyTorch is a CPU-only
build, the selected CUDA device is invalid, or an actual asynchronous
H2D → CUDA kernel → D2H smoke test fails. It therefore cannot silently report
an alleged GPU run that was executed on the CPU. Use `--cuda-device N` when a
runtime exposes more than one GPU. The benchmark itself uses CUDA-enabled
PyTorch rather than a custom `.cu` extension; its GPU path uses a CUDA stream,
pinned host buffers, non-blocking copies, CUDA events, and GPU tensor kernels.

## Output

Every invocation creates `results/run_YYYYMMDD_HHMMSS/` with:

- `environment.json`: CPU/GPU, CUDA, driver, and PyTorch inventory.
- `cuda_preflight.json`: proof that the selected CUDA device completed an
  asynchronous transfer-and-kernel smoke test before the benchmark.
- `config.json`: exact command configuration.
- `trace_<mode>_trial_<n>.csv`: per-task queue, preparation, transfer,
  compute, and end-to-end timestamps.
- `resources_<mode>_trial_<n>.csv`: best-effort process CPU and GPU samples.
- `summary.json` and `summary.csv`: latency percentiles, throughput, transfer
  bytes, correctness status, and across-trial variability.

The outputs include H2D and D2H data movement rather than hiding it in GPU
compute time. Task inputs are reconstructed from deterministic seeds; the
first `--validate-tasks` tasks in every trial are compared with a CPU oracle.

## Tuning guidance

Start with the quick test, then hold workload and input size fixed while
varying one setting at a time:

```bash
!python3 run_benchmark.py --mode all --tasks 40 --batch-size 2048 --width 1024 --gpu-inflight 1 --trials 5
!python3 run_benchmark.py --mode all --tasks 40 --batch-size 8192 --width 1024 --gpu-inflight 2 --trials 5
!python3 run_benchmark.py --mode all --tasks 40 --batch-size 16384 --width 1024 --gpu-inflight 3 --trials 5
```

Also explore `--cpu-workers`, `--static-cpu-share`, and
`--min-gpu-batch`. `--cpu-threads 1` is the default to prevent nested
PyTorch threading from oversubscribing the Colab VM when there are several CPU
workers.

Interpret the hybrid result cautiously: compare the median *throughput* and
p95 *latency* with both CPU-only and GPU-only results, not just a single
trial. GPU utilization is sampled with `nvidia-smi` when available; sampling
is deliberately outside the timed execution path.

## Design notes

- The GPU executor owns a CUDA stream and uses bounded in-flight pinned host
  buffers. It does not synchronize the entire device for every task.
- The `TaskGraph` represents task dependencies; `--dependency-chain` is a
  reproducible synthetic DAG case. Independent work is the appropriate mode
  for testing CPU/GPU overlap.
- Dynamic scheduling learns a rolling time-per-work-unit estimate after a
  short CPU/GPU calibration and considers predicted queue delay. The
  hysteresis prevents moving work for negligible expected gains.
- Outputs are copied back deliberately so the benchmark measures an
  end-to-end pipeline. If a real downstream stage stays on the GPU, adapting
  it to retain GPU-resident output can change the conclusion; measure that
  pipeline separately.
- Float32 is the default and TensorFloat-32 is disabled for tight CPU/GPU
  correctness comparison. Add `--allow-tf32` on supported GPUs when measuring
  the maximum approximate-FP32 matrix throughput instead.

## Local checks

```bash
python3 -m unittest discover -s tests -v
python3 run_benchmark.py --mode cpu --tasks 3 --batch-size 16 --width 8 --trials 1 --warmup 0
```
# hybridbit-research
