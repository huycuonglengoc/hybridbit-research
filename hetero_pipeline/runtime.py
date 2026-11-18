"""Task graph, executors, scheduling policies, and benchmark orchestration."""

from __future__ import annotations

import concurrent.futures
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from .metrics import ResourceSampler, percentile, resource_summary
from .workloads import TensorTree, Workload, tree_pin_memory, tree_to_device


DeviceName = Literal["cpu", "gpu"]
ModeName = Literal["cpu", "gpu", "hybrid-static", "hybrid-dynamic"]


@dataclass(frozen=True)
class Task:
    task_id: int
    workload_id: str
    operation_class: str
    input_bytes: int
    output_bytes: int
    batch_id: int
    batch_size: int
    dependencies: tuple[int, ...]
    allowed_devices: tuple[DeviceName, ...]
    seed: int
    work_units: int
    item_count: int
    source_start: int


class TaskGraph:
    """Minimal DAG model; ready tasks are those with all inputs completed."""

    def __init__(self, tasks: list[Task]) -> None:
        self.tasks = {task.task_id: task for task in tasks}
        if len(self.tasks) != len(tasks):
            raise ValueError("task IDs must be unique")
        for task in tasks:
            unknown = set(task.dependencies) - self.tasks.keys()
            if unknown:
                raise ValueError(f"Task {task.task_id} references missing dependencies: {unknown}")
        self._check_acyclic()

    def _check_acyclic(self) -> None:
        remaining = {task_id: set(task.dependencies) for task_id, task in self.tasks.items()}
        completed: set[int] = set()
        while remaining:
            ready = [task_id for task_id, deps in remaining.items() if deps <= completed]
            if not ready:
                raise ValueError("task graph contains a dependency cycle")
            for task_id in ready:
                completed.add(task_id)
                del remaining[task_id]

    def ready(self, completed: set[int], scheduled: set[int]) -> list[Task]:
        return [
            task
            for task_id, task in sorted(self.tasks.items())
            if task_id not in completed and task_id not in scheduled and set(task.dependencies) <= completed
        ]


@dataclass
class TaskResult:
    task: Task
    executor: str
    enqueue_time: float
    start_time: float
    finish_time: float
    host_prepare_ms: float
    compute_ms: float
    h2d_ms: float = 0.0
    d2h_ms: float = 0.0
    predicted_ms: float = 0.0
    fallback: bool = False
    output: torch.Tensor | None = field(default=None, repr=False)

    @property
    def queue_wait_ms(self) -> float:
        return (self.start_time - self.enqueue_time) * 1000

    @property
    def total_ms(self) -> float:
        return (self.finish_time - self.enqueue_time) * 1000

    def trace_row(self, origin: float) -> dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "batch_id": self.task.batch_id,
            "executor": self.executor,
            "fallback": self.fallback,
            "dependencies": ";".join(map(str, self.task.dependencies)),
            "input_bytes": self.task.input_bytes,
            "output_bytes": self.task.output_bytes,
            "item_count": self.task.item_count,
            "source_start": self.task.source_start,
            "enqueue_ms": (self.enqueue_time - origin) * 1000,
            "start_ms": (self.start_time - origin) * 1000,
            "finish_ms": (self.finish_time - origin) * 1000,
            "queue_wait_ms": self.queue_wait_ms,
            "total_latency_ms": self.total_ms,
            "host_prepare_ms": self.host_prepare_ms,
            "h2d_ms": self.h2d_ms,
            "compute_ms": self.compute_ms,
            "d2h_ms": self.d2h_ms,
            "predicted_ms": self.predicted_ms,
        }


@dataclass
class BenchmarkConfig:
    workload: Workload
    task_count: int = 24
    batch_size: int = 8192
    total_items: int | None = None
    cpu_workers: int = 2
    cpu_threads: int = 1
    gpu_inflight: int = 2
    static_cpu_share: float = 0.25
    min_gpu_batch: int = 2048
    hysteresis_percent: float = 8.0
    seed: int = 1234
    dependency_chain: bool = False
    validate_tasks: int = 3
    pin_memory: bool = True
    gpu_fallback: bool = True
    cuda_device: int = 0
    allow_tf32: bool = False
    counter_start: int = 1

    def validate(self) -> None:
        if self.task_count < 1 or self.batch_size < 1:
            raise ValueError("task_count and batch_size must be positive")
        if self.total_items is not None and self.total_items < 1:
            raise ValueError("total_items must be positive when set")
        if self.cpu_workers < 1 or self.cpu_threads < 1 or self.gpu_inflight < 1:
            raise ValueError("worker, thread, and inflight counts must be positive")
        if not 0 <= self.static_cpu_share <= 1:
            raise ValueError("static_cpu_share must be in [0, 1]")
        if not 0 <= self.hysteresis_percent < 100:
            raise ValueError("hysteresis_percent must be in [0, 100)")
        if self.cuda_device < 0:
            raise ValueError("cuda_device must be non-negative")


class CostModel:
    """EWMA cost model expressed in milliseconds per workload unit."""

    def __init__(self) -> None:
        self._per_unit_ms: dict[str, float] = {}

    def update(self, device: DeviceName, result: TaskResult) -> None:
        # GPU time includes explicit transfers because placement decisions are
        # end-to-end decisions. CPU task latency similarly includes preparation.
        elapsed = result.compute_ms + result.host_prepare_ms + result.h2d_ms + result.d2h_ms
        sample = max(elapsed / max(result.task.work_units, 1), 1e-12)
        previous = self._per_unit_ms.get(device)
        self._per_unit_ms[device] = sample if previous is None else previous * 0.7 + sample * 0.3

    def estimate(self, device: DeviceName, task: Task) -> float:
        baseline = self._per_unit_ms.get(device)
        if baseline is None:
            # A conservative prior: GPU generally has a lower compute cost but
            # nontrivial fixed transfer/launch overhead.
            return (0.002 if device == "cpu" else 0.0008) * task.work_units + (0 if device == "cpu" else 0.3)
        return max(0.01, baseline * task.work_units)

    def snapshot(self) -> dict[str, float]:
        return {f"{device}_ms_per_unit": value for device, value in self._per_unit_ms.items()}


def _output_digest(value: torch.Tensor) -> None:
    # Forces no device synchronization for CPU. The result is intentionally
    # unused: it touches output before reusable buffers are released.
    if value.numel() == 0:
        return
    _ = float(value.reshape(-1)[0])


class CPUExecutor:
    def __init__(self, workload: Workload, workers: int) -> None:
        self.workload = workload
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cpu-worker")

    def submit(self, task: Task, keep_output: bool, predicted_ms: float, fallback: bool = False) -> concurrent.futures.Future[TaskResult]:
        enqueued = time.perf_counter()
        return self.pool.submit(self._run, task, enqueued, keep_output, predicted_ms, fallback)

    def _run(
        self, task: Task, enqueued: float, keep_output: bool, predicted_ms: float, fallback: bool
    ) -> TaskResult:
        start = time.perf_counter()
        prepare_start = start
        value = self.workload.make_input(task.seed, task.batch_size)
        prepared = time.perf_counter()
        with torch.inference_mode():
            compute_start = time.perf_counter()
            output = self.workload.run(value)
            _output_digest(output)
            finish = time.perf_counter()
        result_output = output.detach().cpu() if keep_output else None
        return TaskResult(
            task=task,
            executor="cpu",
            enqueue_time=enqueued,
            start_time=start,
            finish_time=finish,
            host_prepare_ms=(prepared - prepare_start) * 1000,
            compute_ms=(finish - compute_start) * 1000,
            predicted_ms=predicted_ms,
            fallback=fallback,
            output=result_output,
        )

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=False)


@dataclass
class GPUTicket:
    task: Task
    enqueue_time: float
    start_time: float
    host_prepare_ms: float
    h2d_start: torch.cuda.Event
    h2d_end: torch.cuda.Event
    compute_start: torch.cuda.Event
    compute_end: torch.cuda.Event
    d2h_start: torch.cuda.Event
    d2h_end: torch.cuda.Event
    done: torch.cuda.Event
    # Keep both sides of asynchronous copies alive until the completion event.
    # This is required for pinned input/output buffers used by CUDA DMA.
    input_host: TensorTree
    input_device: TensorTree
    output_device: torch.Tensor
    output_host: torch.Tensor
    predicted_ms: float


class GPUExecutor:
    """Single CUDA stream with bounded outstanding buffers.

    Work is submitted asynchronously.  The runner polls completion events
    while CPU workers continue processing their own batches.
    """

    def __init__(self, workload: Workload, pin_memory: bool, cuda_device: int) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if cuda_device >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {cuda_device} is not visible")
        self.workload = workload
        self.pin_memory = pin_memory
        self.device = torch.device(f"cuda:{cuda_device}")
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.Stream(device=self.device)
        self._output_pool: dict[tuple[tuple[int, ...], torch.dtype], list[torch.Tensor]] = defaultdict(list)

    def _acquire_output_buffer(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Reuse completed pinned output buffers instead of growing per task."""
        available = self._output_pool[(shape, dtype)]
        if available:
            return available.pop()
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=self.pin_memory)

    def _release_output_buffer(self, output: torch.Tensor) -> None:
        self._output_pool[(tuple(output.shape), output.dtype)].append(output)

    def submit(self, task: Task, keep_output: bool, predicted_ms: float) -> GPUTicket:
        enqueued = time.perf_counter()
        start = enqueued
        value: TensorTree = self.workload.make_input(task.seed, task.batch_size)
        if self.pin_memory:
            value = tree_pin_memory(value)
        prepared = time.perf_counter()
        output_host = self._acquire_output_buffer(
            self.workload.output_shape(task.batch_size), self.workload.output_dtype()
        )
        h2d_start, h2d_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        compute_start, compute_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        d2h_start, d2h_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        done = torch.cuda.Event(enable_timing=False)
        with torch.cuda.stream(self.stream), torch.inference_mode():
            h2d_start.record(self.stream)
            gpu_value = tree_to_device(value, self.device, non_blocking=self.pin_memory)
            h2d_end.record(self.stream)
            compute_start.record(self.stream)
            gpu_output = self.workload.run(gpu_value)
            compute_end.record(self.stream)
            d2h_start.record(self.stream)
            output_host.copy_(gpu_output, non_blocking=self.pin_memory)
            d2h_end.record(self.stream)
            done.record(self.stream)
        return GPUTicket(
            task=task,
            enqueue_time=enqueued,
            start_time=start,
            host_prepare_ms=(prepared - start) * 1000,
            h2d_start=h2d_start,
            h2d_end=h2d_end,
            compute_start=compute_start,
            compute_end=compute_end,
            d2h_start=d2h_start,
            d2h_end=d2h_end,
            done=done,
            input_host=value,
            input_device=gpu_value,
            output_device=gpu_output,
            output_host=output_host,
            predicted_ms=predicted_ms,
        )

    @staticmethod
    def is_ready(ticket: GPUTicket) -> bool:
        return ticket.done.query()

    def collect(self, ticket: GPUTicket, keep_output: bool) -> TaskResult:
        ticket.done.synchronize()
        finish = time.perf_counter()
        output = ticket.output_host
        _output_digest(output)
        # The buffer returns to the pool immediately. Validation retains only a
        # small explicit clone, so it cannot race the next asynchronous copy.
        result_output = output.clone() if keep_output else None
        result = TaskResult(
            task=ticket.task,
            executor="gpu",
            enqueue_time=ticket.enqueue_time,
            start_time=ticket.start_time,
            finish_time=finish,
            host_prepare_ms=ticket.host_prepare_ms,
            h2d_ms=ticket.h2d_start.elapsed_time(ticket.h2d_end),
            compute_ms=ticket.compute_start.elapsed_time(ticket.compute_end),
            d2h_ms=ticket.d2h_start.elapsed_time(ticket.d2h_end),
            predicted_ms=ticket.predicted_ms,
            output=result_output,
        )
        self._release_output_buffer(output)
        return result


@dataclass
class TrialResult:
    mode: str
    trial_index: int
    started_at: float
    finished_at: float
    tasks: list[TaskResult]
    resource_samples: list[Any]
    validation_errors: list[str]
    cost_model: dict[str, float]

    def summary(self) -> dict[str, Any]:
        duration_s = self.finished_at - self.started_at
        latencies = [task.total_ms for task in self.tasks]
        item_count = sum(task.task.item_count for task in self.tasks)
        per_executor: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for task in self.tasks:
            bucket = per_executor[task.executor]
            bucket["tasks"] += 1
            bucket["compute_ms"] += task.compute_ms
            bucket["h2d_ms"] += task.h2d_ms
            bucket["d2h_ms"] += task.d2h_ms
            bucket["input_bytes"] += task.task.input_bytes if task.executor == "gpu" else 0
            bucket["output_bytes"] += task.task.output_bytes if task.executor == "gpu" else 0
        return {
            "mode": self.mode,
            "trial_index": self.trial_index,
            "duration_s": duration_s,
            "task_count": len(self.tasks),
            "throughput_tasks_per_s": len(self.tasks) / duration_s if duration_s else 0.0,
            "item_count": item_count,
            "items_per_second": item_count / duration_s if duration_s else 0.0,
            "latency_ms": {
                "mean": statistics.fmean(latencies) if latencies else 0.0,
                "median": percentile(latencies, 0.5),
                "p95": percentile(latencies, 0.95),
                "p99": percentile(latencies, 0.99),
            },
            "per_executor": {key: dict(value) for key, value in per_executor.items()},
            "transfer_bytes": sum(
                task.task.input_bytes + task.task.output_bytes for task in self.tasks if task.executor == "gpu"
            ),
            "validation_passed": not self.validation_errors,
            "validation_errors": self.validation_errors,
            "resource": resource_summary(self.resource_samples),
            "cost_model": self.cost_model,
        }


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig) -> None:
        config.validate()
        self.config = config
        # Avoid TensorFloat-32 changes that would make validation thresholds
        # conceal a CPU/GPU numerical mismatch on Colab GPUs.
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
            torch.backends.cudnn.allow_tf32 = config.allow_tf32
        torch.set_num_threads(config.cpu_threads)

    @property
    def has_gpu(self) -> bool:
        return torch.cuda.is_available() and self.config.cuda_device < torch.cuda.device_count()

    def make_tasks(self) -> list[Task]:
        tasks: list[Task] = []
        remaining = self.config.total_items
        index = 0
        emitted_items = 0
        while remaining is None or remaining > 0:
            if remaining is None and index >= self.config.task_count:
                break
            task_batch_size = self.config.batch_size if remaining is None else min(self.config.batch_size, remaining)
            shape = self.config.workload.shape_for(task_batch_size)
            dependencies = (index - 1,) if self.config.dependency_chain and index else ()
            is_counter_workload = self.config.workload.name == "multiword"
            input_seed = (
                self.config.counter_start + emitted_items
                if is_counter_workload
                else self.config.seed + index * 7919
            )
            tasks.append(
                Task(
                    task_id=index,
                    workload_id=self.config.workload.name,
                    operation_class=self.config.workload.name,
                    input_bytes=shape.input_bytes,
                    output_bytes=shape.output_bytes,
                    batch_id=index,
                    batch_size=task_batch_size,
                    dependencies=dependencies,
                    allowed_devices=("cpu", "gpu") if self.has_gpu else ("cpu",),
                    seed=input_seed,
                    work_units=shape.work_units,
                    item_count=self.config.workload.items_in_batch(task_batch_size),
                    source_start=input_seed,
                )
            )
            emitted_items += self.config.workload.items_in_batch(task_batch_size)
            if remaining is not None:
                remaining -= task_batch_size
            index += 1
        return tasks

    def _calibrate(self, cpu: CPUExecutor, gpu: GPUExecutor, model: CostModel) -> None:
        task = self.make_tasks()[0]
        cpu_result = cpu.submit(task, keep_output=False, predicted_ms=0).result()
        model.update("cpu", cpu_result)
        ticket = gpu.submit(task, keep_output=False, predicted_ms=0)
        gpu_result = gpu.collect(ticket, keep_output=False)
        model.update("gpu", gpu_result)

    def _static_device(self, task_id: int) -> DeviceName:
        # Bresenham-like allocation spreads CPU tasks across the task stream.
        current = math.floor((task_id + 1) * self.config.static_cpu_share)
        previous = math.floor(task_id * self.config.static_cpu_share)
        return "cpu" if current > previous else "gpu"

    def _choose_dynamic(
        self, task: Task, model: CostModel, cpu_load_ms: float, gpu_load_ms: float
    ) -> tuple[DeviceName, float]:
        cpu_prediction = model.estimate("cpu", task) + cpu_load_ms / self.config.cpu_workers
        if task.batch_size < self.config.min_gpu_batch:
            return "cpu", cpu_prediction
        gpu_prediction = model.estimate("gpu", task) + gpu_load_ms
        margin = 1 - self.config.hysteresis_percent / 100
        if gpu_prediction < cpu_prediction * margin:
            return "gpu", gpu_prediction
        return "cpu", cpu_prediction

    def _validate(self, results: list[TaskResult]) -> list[str]:
        errors: list[str] = []
        for result in results:
            if result.output is None:
                continue
            value = self.config.workload.make_input(result.task.seed, result.task.batch_size)
            with torch.inference_mode():
                expected = self.config.workload.run(value)
            if result.output.dtype.is_floating_point:
                matches = torch.allclose(result.output, expected, rtol=2e-4, atol=2e-5)
            else:
                matches = torch.equal(result.output, expected)
            if not matches:
                max_error = (result.output - expected).abs().max().item()
                errors.append(f"task {result.task.task_id}: maximum absolute error {max_error:.6g}")
        return errors

    def run(self, mode: ModeName, trial_index: int = 1) -> TrialResult:
        if mode not in ("cpu", "gpu", "hybrid-static", "hybrid-dynamic"):
            raise ValueError(f"Unsupported mode: {mode}")
        if mode != "cpu" and not self.has_gpu:
            raise RuntimeError("GPU mode requested, but CUDA is unavailable")

        needs_cpu = mode in ("cpu", "hybrid-static", "hybrid-dynamic")
        needs_gpu = mode in ("gpu", "hybrid-static", "hybrid-dynamic")
        cpu = CPUExecutor(self.config.workload, self.config.cpu_workers) if needs_cpu else None
        gpu = GPUExecutor(self.config.workload, self.config.pin_memory, self.config.cuda_device) if needs_gpu else None
        model = CostModel()
        if mode == "hybrid-dynamic":
            assert cpu is not None and gpu is not None
            self._calibrate(cpu, gpu, model)

        graph = TaskGraph(self.make_tasks())
        scheduled: set[int] = set()
        completed: set[int] = set()
        results: list[TaskResult] = []
        cpu_futures: dict[concurrent.futures.Future[TaskResult], float] = {}
        gpu_tickets: dict[int, tuple[GPUTicket, float]] = {}
        cpu_load_ms, gpu_load_ms = 0.0, 0.0
        sampler = ResourceSampler()
        started = time.perf_counter()
        sampler.start()

        def should_keep(task: Task) -> bool:
            return task.task_id < self.config.validate_tasks

        def submit_cpu(task: Task, prediction: float, fallback: bool = False) -> None:
            nonlocal cpu_load_ms
            assert cpu is not None
            future = cpu.submit(task, should_keep(task), prediction, fallback)
            cpu_futures[future] = prediction
            cpu_load_ms += prediction
            scheduled.add(task.task_id)

        def submit_gpu(task: Task, prediction: float) -> None:
            nonlocal gpu_load_ms
            assert gpu is not None
            ticket = gpu.submit(task, should_keep(task), prediction)
            gpu_tickets[task.task_id] = (ticket, prediction)
            gpu_load_ms += prediction
            scheduled.add(task.task_id)

        try:
            while len(completed) < len(graph.tasks):
                made_progress = False
                for task in graph.ready(completed, scheduled):
                    # Bound pending work: the bounds make buffer ownership and
                    # peak memory practical for a Colab instance.
                    cpu_has_capacity = cpu is not None and len(cpu_futures) < self.config.cpu_workers * 2
                    gpu_has_capacity = gpu is not None and len(gpu_tickets) < self.config.gpu_inflight
                    if not cpu_has_capacity and not gpu_has_capacity:
                        break

                    if mode == "cpu":
                        device, prediction = "cpu", model.estimate("cpu", task)
                    elif mode == "gpu":
                        device, prediction = "gpu", model.estimate("gpu", task)
                    elif mode == "hybrid-static":
                        device = self._static_device(task.task_id)
                        prediction = model.estimate(device, task)
                    else:
                        device, prediction = self._choose_dynamic(task, model, cpu_load_ms, gpu_load_ms)

                    # Capacity is a hard constraint, whereas the policy is a
                    # preference. Move to the other compatible device when full.
                    if device == "cpu" and not cpu_has_capacity:
                        if not gpu_has_capacity:
                            break
                        device, prediction = "gpu", model.estimate("gpu", task) + gpu_load_ms
                    elif device == "gpu" and not gpu_has_capacity:
                        if not cpu_has_capacity:
                            break
                        device, prediction = "cpu", model.estimate("cpu", task) + cpu_load_ms / self.config.cpu_workers

                    try:
                        if device == "cpu":
                            submit_cpu(task, prediction)
                        else:
                            submit_gpu(task, prediction)
                    except RuntimeError:
                        if device == "gpu" and self.config.gpu_fallback and cpu is not None:
                            submit_cpu(task, model.estimate("cpu", task), fallback=True)
                        else:
                            raise
                    made_progress = True

                # First collect everything already complete without blocking.
                for future in [future for future in cpu_futures if future.done()]:
                    prediction = cpu_futures.pop(future)
                    result = future.result()
                    cpu_load_ms = max(0.0, cpu_load_ms - prediction)
                    model.update("cpu", result)
                    completed.add(result.task.task_id)
                    results.append(result)
                    made_progress = True
                for task_id, (ticket, prediction) in list(gpu_tickets.items()):
                    if not GPUExecutor.is_ready(ticket):
                        continue
                    del gpu_tickets[task_id]
                    try:
                        assert gpu is not None
                        result = gpu.collect(ticket, should_keep(ticket.task))
                    except RuntimeError:
                        if not self.config.gpu_fallback or cpu is None:
                            raise
                        gpu_load_ms = max(0.0, gpu_load_ms - prediction)
                        scheduled.remove(task_id)
                        submit_cpu(ticket.task, model.estimate("cpu", ticket.task), fallback=True)
                        continue
                    gpu_load_ms = max(0.0, gpu_load_ms - prediction)
                    model.update("gpu", result)
                    completed.add(result.task.task_id)
                    results.append(result)
                    made_progress = True

                if made_progress:
                    continue
                if cpu_futures:
                    concurrent.futures.wait(cpu_futures, timeout=0.002, return_when=concurrent.futures.FIRST_COMPLETED)
                elif gpu_tickets:
                    # Event synchronization happens only when no CPU work is
                    # outstanding; this retains CPU/GPU overlap in hybrid mode.
                    first_ticket = next(iter(gpu_tickets.values()))[0]
                    first_ticket.done.synchronize()
                else:
                    raise RuntimeError("scheduler made no progress with no pending work")
        finally:
            samples = sampler.stop()
            if cpu is not None:
                cpu.close()

        finished = time.perf_counter()
        results.sort(key=lambda result: result.task.task_id)
        validation_errors = self._validate(results)
        return TrialResult(
            mode=mode,
            trial_index=trial_index,
            started_at=started,
            finished_at=finished,
            tasks=results,
            resource_samples=samples,
            validation_errors=validation_errors,
            cost_model=model.snapshot(),
        )
