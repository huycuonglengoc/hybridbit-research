"""Trace export, timing summaries, hardware inventory, and resource samples."""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _cpu_model() -> str:
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except (OSError, subprocess.SubprocessError):
            return platform.processor() or "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _nvidia_smi(fields: str) -> list[str] | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return [line.strip() for line in output.splitlines() if line.strip()]


def cuda_preflight(device_index: int = 0) -> dict[str, Any]:
    """Prove that this interpreter can submit a real CUDA kernel.

    ``torch.cuda.is_available()`` alone is not sufficient: a Colab kernel can
    have an NVIDIA driver visible while using a CPU-only PyTorch wheel.  The
    smoke test below creates a non-default CUDA stream, performs pinned H2D
    and D2H copies, launches a CUDA operation, and validates its result.
    """
    report: dict[str, Any] = {
        "requested_device": device_index,
        "pytorch": torch.__version__,
        "pytorch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "ok": False,
        "errors": [],
    }
    if device_index < 0:
        report["errors"].append("CUDA device index must be non-negative")
        return report
    if torch.version.cuda is None:
        report["errors"].append("Installed PyTorch was built without CUDA support")
        return report
    if not torch.cuda.is_available():
        report["errors"].append("PyTorch cannot initialize CUDA; enable a GPU runtime in Colab")
        return report

    count = torch.cuda.device_count()
    report["device_count"] = count
    if device_index >= count:
        report["errors"].append(f"Requested CUDA device {device_index}, but only {count} device(s) are visible")
        return report

    try:
        device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        report["device"] = {
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": properties.total_memory,
            "free_memory_bytes": free_bytes,
            "reported_total_memory_bytes": total_bytes,
        }
        report["cudnn_version"] = torch.backends.cudnn.version()
        report["compiled_arches"] = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []

        # The temporary buffers deliberately exercise the same asynchronous
        # data path used by GPUExecutor, not merely CUDA device discovery.
        source = torch.arange(1024, dtype=torch.float32, pin_memory=True)
        destination = torch.empty_like(source, pin_memory=True)
        stream = torch.cuda.Stream(device=device)
        done = torch.cuda.Event(enable_timing=False)
        with torch.cuda.stream(stream):
            device_source = source.to(device, non_blocking=True)
            device_result = device_source * 2.0 + 1.0
            destination.copy_(device_result, non_blocking=True)
            done.record(stream)
        done.synchronize()
        if not torch.equal(destination, source * 2.0 + 1.0):
            raise RuntimeError("CUDA smoke-test output did not match the CPU reference")
        report["kernel_smoke_test"] = "passed"
        report["async_copy_smoke_test"] = "passed"
        report["ok"] = True
    except Exception as error:  # report the setup problem without a traceback wall
        report["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            # The original initialization error is more useful than a cache
            # cleanup failure on a broken CUDA runtime.
            pass
    return report


def environment_report(device_index: int = 0) -> dict[str, Any]:
    try:
        physical_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        physical_memory = None
    report: dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "cpu_logical_cores": os.cpu_count(),
        "physical_memory_bytes": physical_memory,
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
    }
    if torch.cuda.is_available() and 0 <= device_index < torch.cuda.device_count():
        report["gpu"] = {
            "selected_device": device_index,
            "name": torch.cuda.get_device_name(device_index),
            "capability": list(torch.cuda.get_device_capability(device_index)),
            "total_memory_bytes": torch.cuda.get_device_properties(device_index).total_memory,
        }
    nvidia = _nvidia_smi("name,driver_version,memory.total")
    if nvidia:
        report["nvidia_smi"] = nvidia
    return report


@dataclass
class ResourceSample:
    timestamp_s: float
    cpu_process_percent: float | None
    gpu_utilization_percent: float | None
    gpu_memory_used_mib: float | None


class ResourceSampler:
    """Best-effort, low-frequency utilization sampling outside the hot path."""

    def __init__(self, interval_s: float = 0.25) -> None:
        self.interval_s = interval_s
        self.samples: list[ResourceSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        try:
            import psutil  # type: ignore

            self._process = psutil.Process()
        except ImportError:
            self._process = None

    def start(self) -> None:
        self._started_at = time.perf_counter()
        if self._process is not None:
            self._process.cpu_percent(None)
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            cpu = self._process.cpu_percent(None) if self._process is not None else None
            gpu_util: float | None = None
            gpu_mem: float | None = None
            nvidia = _nvidia_smi("utilization.gpu,memory.used")
            if nvidia:
                try:
                    pieces = [piece.strip() for piece in nvidia[0].split(",")]
                    gpu_util, gpu_mem = float(pieces[0]), float(pieces[1])
                except (ValueError, IndexError):
                    pass
            self.samples.append(ResourceSample(time.perf_counter() - self._started_at, cpu, gpu_util, gpu_mem))

    def stop(self) -> list[ResourceSample]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 1)
        return self.samples


def resource_summary(samples: Iterable[ResourceSample]) -> dict[str, float | None]:
    values = list(samples)
    cpu = [sample.cpu_process_percent for sample in values if sample.cpu_process_percent is not None]
    gpu = [sample.gpu_utilization_percent for sample in values if sample.gpu_utilization_percent is not None]
    memory = [sample.gpu_memory_used_mib for sample in values if sample.gpu_memory_used_mib is not None]
    return {
        "sample_count": len(values),
        "avg_process_cpu_percent": statistics.fmean(cpu) if cpu else None,
        "avg_gpu_utilization_percent": statistics.fmean(gpu) if gpu else None,
        "max_gpu_memory_used_mib": max(memory) if memory else None,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def export_samples(path: Path, samples: list[ResourceSample]) -> None:
    write_csv(path, [asdict(sample) for sample in samples])
