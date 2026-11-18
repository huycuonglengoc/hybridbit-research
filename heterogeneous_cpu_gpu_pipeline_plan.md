# Implementation Plan: CPU--GPU Heterogeneous Compute Pipeline

## 1. Goal

Build a reusable research framework that evaluates and runs independent
or dependency-linked numeric workloads across CPU and GPU. The framework
should:

-   Keep CPU and GPU busy where practical.
-   Use batching to amortize launch, transfer, and scheduling overhead.
-   Measure end-to-end throughput, latency, utilization, and transfer
    costs.
-   Compare CPU-only, GPU-only, and hybrid execution.
-   Avoid assumptions that a given operation is always faster on one
    device.

This plan is for a **general numeric workload framework**. Use
non-cryptographic test workloads when validating scheduling behavior.

## 2. Non-goals

-   Do not assume CPU/GPU splitting is faster than a single-device
    implementation.
-   Do not optimize around isolated microbenchmarks alone.
-   Do not make device selection solely from operation labels (e.g.,
    "multiply = GPU").
-   Do not add specialized cryptographic key-generation or
    address-generation paths to this framework.

## 3. Architecture

``` text
                 ┌────────────────────┐
                 │ Workload Producer │
                 └─────────┬────────┘
                           │ task batches
                 ┌─────────▼─────────┐
                 │ Task / DAG Queue  │
                 └─────────┬─────────┘
                           │
                 ┌─────────▼─────────┐
                 │ Cost Model &      │
                 │ Scheduler         │
                 └──────┬─────┬──────┘
                        │     │
               CPU path│     │GPU path
             ┌────────▼───┐ ┌▼────────────────┐
             │ CPU Worker │ │ GPU Executor    │
             │ Pool       │ │ (async batches) │
             └──────┬─────┘ └┬────────────────┘
                    │        │
                    └────┬───┘
                         │ completion events
                 ┌───────▼─────────┐
                 │ Result Collector│
                 └───────┬─────────┘
                         │
                 ┌───────▼─────────┐
                 │ Metrics / Trace│
                 └────────────────┘
```

### Components

1.  **Workload Producer** --- generates tasks and batch metadata.
2.  **Task/DAG Queue** --- represents tasks and dependencies; ready
    tasks can be scheduled.
3.  **Cost Model** --- estimates CPU time, GPU time, transfer cost,
    queue delay, and synchronization overhead.
4.  **Scheduler** --- selects an executor, subject to dependencies and
    buffer availability.
5.  **CPU Worker Pool** --- executes CPU-compatible tasks.
6.  **GPU Executor** --- submits batches asynchronously and tracks
    completion events.
7.  **Result Collector** --- joins completed tasks, validates outputs,
    and releases buffers.
8.  **Metrics/Trace** --- records timings and counters for analysis.

## 4. Task model

Represent each task with metadata rather than hard-coding device
assignment.

Suggested fields:

-   `task_id`: unique identifier
-   `workload_id`: benchmark or experiment name
-   `operation_class`: generic category (e.g., integer arithmetic,
    transform, reduction)
-   `input_bytes`, `output_bytes`
-   `batch_id`
-   `dependencies`: IDs of prerequisite tasks
-   `allowed_devices`: CPU, GPU, or either
-   `estimated_cost_cpu`, `estimated_cost_gpu` (optional, learned from
    measurements)
-   `deadline` or priority (optional)
-   `state`: queued, ready, running, completed, failed
-   `timestamps`: enqueue, start, finish

Use a DAG when tasks have dependencies. A task is ready only after all
dependencies have completed and its input data is available.

## 5. Scheduling policy

Start with simple policies and add sophistication only after
measurements.

### Phase A --- Baselines

Implement: - CPU-only FIFO - GPU-only batching - Static split (fixed
share between CPU and GPU) - Hybrid work-stealing for independent tasks

### Phase B --- Measurement-based assignment

For each candidate device, estimate:

`predicted_total = queue_wait + compute_time + transfer_time + sync_cost`

Assign a task only when the estimated total cost is lower and the task
is compatible with the device. Include a safety margin to avoid
oscillation from noisy measurements.

### Phase C --- Adaptive policy

-   Update cost estimates using rolling measurements.
-   Keep a minimum batch size and maximum wait time.
-   Avoid repeatedly migrating data between devices.
-   Prefer executing a dependency chain where its working data already
    resides.
-   Add hysteresis: do not change assignment for tiny predicted
    improvements.

Do not introduce a complex scheduler until the baselines are reliable.

## 6. Batching and pipeline behavior

Use bounded queues and double/triple buffering to avoid unbounded memory
growth.

Each batch should track: - batch size - input/output buffer ownership -
device residency - enqueue and completion events - timeout/error state

Tune batch size empirically. Test small, medium, and large batches;
record both throughput and tail latency. A large batch may improve
throughput but increase latency and memory pressure.

Use asynchronous transfers and execution where the platform supports
them. Ensure buffers remain valid until the associated transfer or
kernel has completed. Avoid synchronizing the entire device after every
task; synchronize at dependency or measurement boundaries.

## 7. Data movement

Treat data movement as a first-class cost.

-   Keep data on its current device when subsequent tasks can use it
    there.
-   Transfer only when required by a dependency or final result.
-   Reuse pinned host buffers where supported and justified by
    measurements.
-   Track bytes transferred in each direction.
-   Separate transfer time from compute time in metrics.
-   Avoid tiny transfers; aggregate compatible work.

## 8. Workload suite

Use safe, generic numeric workloads to exercise different
characteristics:

  -----------------------------------------------------------------------
  Workload                Characteristics         What it tests
  ----------------------- ----------------------- -----------------------
  Elementwise integer     independent items       throughput and batching
  arithmetic                                      

  Large-integer           multiword arithmetic    CPU/GPU implementation
  arithmetic test vectors                         differences

  Matrix/tensor           high parallelism        GPU utilization
  operations                                      

  Reduction               shared aggregation      memory bandwidth and
                                                  synchronization

  Dependency-chain        serial dependencies     scheduling and transfer
  synthetic DAG                                   overhead

  Memory-copy test        transfer-only           transfer ceiling and
                                                  buffer behavior
  -----------------------------------------------------------------------

Every workload must have a correctness oracle and deterministic test
inputs.

## 9. Benchmark methodology

Compare the following modes on identical inputs:

1.  CPU-only
2.  GPU-only
3.  Static hybrid split
4.  Dynamic hybrid scheduler

Record: - tasks per second - batch latency (median, p95, p99) - compute
time per device - transfer time and bytes - queue wait time - CPU
utilization - GPU utilization - memory use - energy, if available

Warm up the device and runtime before measuring. Run multiple trials,
report median and variability, and avoid comparing runs with different
power, thermal, or background-load conditions.

## 10. Correctness and reliability

-   Unit-test task state transitions and dependency handling.
-   Compare CPU and GPU outputs against a trusted reference for each
    generic workload.
-   Test empty batches, boundary sizes, overflow behavior, and invalid
    inputs.
-   Test GPU failure, timeout, CPU fallback, and cancellation.
-   Verify buffer ownership and lifetime under asynchronous execution.
-   Use race detection and memory-checking tools where available.
-   Keep deterministic replay of benchmark traces.

## 11. Observability

Provide a structured trace for each task:

-   task and batch IDs
-   chosen executor
-   estimated vs. measured duration
-   transfer bytes and duration
-   queue wait
-   dependency wait
-   completion status

Expose aggregate counters and export CSV/JSON for analysis. Add a
timeline view showing CPU tasks, GPU submissions, transfers, and idle
gaps.

## 12. Suggested repository structure

``` text
hetero-pipeline/
  README.md
  docs/
    architecture.md
    benchmark-methodology.md
  src/
    workload/
    task_graph/
    scheduler/
    cpu_executor/
    gpu_executor/
    buffers/
    metrics/
    runtime/
  tests/
    unit/
    integration/
    correctness/
    failure/
  benchmarks/
    workloads/
    scripts/
    results/
  config/
    default.yaml
```

## 13. Milestones

### M0 --- Environment inventory

-   Identify CPU model, core count, instruction capabilities, RAM, GPU
    model, VRAM, driver/runtime, and transfer link.
-   Record compiler and library versions.
-   Confirm a minimal CPU and GPU test can run.

**Exit criterion:** environment report and reproducible build.

### M1 --- Correct single-device baselines

-   Implement generic workload interface.
-   Run CPU-only and GPU-only modes.
-   Add correctness checks and benchmark harness.

**Exit criterion:** stable baseline numbers with correctness passing.

### M2 --- Batch and asynchronous execution

-   Add bounded batch queues.
-   Add asynchronous GPU submission and completion tracking.
-   Add reusable buffers and transfer measurements.

**Exit criterion:** no buffer-lifetime errors; benchmark shows
batch-size tradeoffs.

### M3 --- Static hybrid pipeline

-   Implement CPU and GPU executor pools.
-   Split independent generic tasks using configurable proportions.
-   Overlap CPU work with GPU execution.

**Exit criterion:** demonstrate overlap in traces; compare against both
baselines.

### M4 --- DAG scheduling

-   Add task dependencies and ready-task tracking.
-   Add device-residency metadata and transfer accounting.
-   Implement simple cost-based placement.

**Exit criterion:** scheduler decisions and dependency correctness are
replayable.

### M5 --- Adaptive scheduler

-   Update cost estimates from measurements.
-   Add hysteresis and minimum batch thresholds.
-   Test under changing workload sizes.

**Exit criterion:** adaptive mode is not consistently worse than the
best static policy on the test suite.

### M6 --- Hardening and reporting

-   Add failure recovery, timeouts, cancellation, and trace export.
-   Produce a report with workloads, hardware, settings, and
    confidence/variability.

**Exit criterion:** repeatable benchmark package and documented
limitations.

## 14. Configuration sketch

Keep configuration simple and explicit:

``` yaml
runtime:
  mode: hybrid
  cpu_workers: auto
  gpu_enabled: true

scheduler:
  policy: measured_cost
  min_batch_size: configurable
  max_batch_wait_ms: configurable
  hysteresis_percent: configurable

buffers:
  count: 2
  max_inflight_batches: configurable

metrics:
  trace: true
  export: csv
```

Values should be tuned by experiments, not assumed to be universal
defaults.

## 15. Experiment matrix

Run a matrix across: - batch size - CPU worker count - number of
in-flight batches - static CPU/GPU split - transfer strategy - workload
type - scheduler policy

For each configuration, retain raw results and metadata. Avoid selecting
a winner from one run.

## 16. Decision criteria

A hybrid configuration is considered useful only if it: - Improves
throughput over the best single-device baseline for the target
workload. - Meets latency constraints. - Does not cause unacceptable
memory growth. - Remains correct under load and failure conditions. -
Shows repeatable gains across trials.

## 17. Final deliverables

-   Reusable CPU/GPU execution framework
-   Workload interface and DAG model
-   CPU and GPU executors
-   Baseline and hybrid schedulers
-   Benchmark suite and correctness tests
-   Trace and metrics export
-   Reproducible performance report

## 18. First implementation order

1.  Environment inventory
2.  Generic workload interface
3.  CPU baseline
4.  GPU baseline
5.  Benchmark and correctness harness
6.  Batching and buffer reuse
7.  Asynchronous GPU execution
8.  Static hybrid scheduling
9.  Dependency DAG
10. Measurement-based scheduling
11. Adaptive policy only after baseline evidence

**Engineering principle:** optimize the end-to-end pipeline, not the
isolated speed of an operation. A device choice is beneficial only when
compute savings exceed transfer, queueing, and synchronization costs.
