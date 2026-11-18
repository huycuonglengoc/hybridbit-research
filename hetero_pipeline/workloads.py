"""Deterministic numeric workloads used by the benchmark harness.

The workloads are deliberately generic and non-cryptographic.  Each task can
recreate its input from a seed, which keeps correctness checks reproducible
without retaining all input buffers in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Union

import torch


TensorTree = Union[torch.Tensor, tuple[torch.Tensor, ...]]


@dataclass(frozen=True)
class WorkloadShape:
    """Shape and cost metadata for one task."""

    input_bytes: int
    output_bytes: int
    work_units: int


class Workload(Protocol):
    """Interface shared by CPU and GPU executors."""

    name: str

    def shape_for(self, batch_size: int) -> WorkloadShape: ...

    def make_input(self, seed: int, batch_size: int) -> TensorTree: ...

    def run(self, value: TensorTree) -> torch.Tensor: ...

    def output_shape(self, batch_size: int) -> tuple[int, ...]: ...

    def items_in_batch(self, batch_size: int) -> int: ...


def _random_tensor(seed: int, shape: tuple[int, ...]) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    # A bounded input gives consistent numerical tolerances on CPU and GPU.
    return torch.rand(shape, generator=generator, dtype=torch.float32) - 0.5


@dataclass(frozen=True)
class ElementwiseWorkload:
    """High-throughput independent floating-point arithmetic."""

    width: int
    name: str = "elementwise"

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError("width must be positive")

    def shape_for(self, batch_size: int) -> WorkloadShape:
        values = batch_size * self.width
        return WorkloadShape(values * 4, values * 4, values)

    def make_input(self, seed: int, batch_size: int) -> torch.Tensor:
        return _random_tensor(seed, (batch_size, self.width))

    def run(self, value: TensorTree) -> torch.Tensor:
        assert isinstance(value, torch.Tensor)
        # Several independent elementwise ops make launch/transfer overhead
        # visible at small batch sizes, while remaining a generic workload.
        return torch.sin(value) * 1.37 + torch.cos(value * 0.17) + value.square()

    def output_shape(self, batch_size: int) -> tuple[int, ...]:
        return (batch_size, self.width)

    def items_in_batch(self, batch_size: int) -> int:
        return batch_size


@dataclass(frozen=True)
class ReductionWorkload:
    """Memory-bandwidth-oriented row reduction."""

    width: int
    name: str = "reduction"

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError("width must be positive")

    def shape_for(self, batch_size: int) -> WorkloadShape:
        return WorkloadShape(batch_size * self.width * 4, batch_size * 4, batch_size * self.width)

    def make_input(self, seed: int, batch_size: int) -> torch.Tensor:
        return _random_tensor(seed, (batch_size, self.width))

    def run(self, value: TensorTree) -> torch.Tensor:
        assert isinstance(value, torch.Tensor)
        return (value * value + 0.125).sum(dim=1)

    def output_shape(self, batch_size: int) -> tuple[int, ...]:
        return (batch_size,)

    def items_in_batch(self, batch_size: int) -> int:
        return batch_size


@dataclass(frozen=True)
class MatmulWorkload:
    """Dense matrix multiplication, selected to exercise GPU compute."""

    matrix_dim: int
    name: str = "matmul"

    def __post_init__(self) -> None:
        if self.matrix_dim < 1:
            raise ValueError("matrix_dim must be positive")

    def shape_for(self, batch_size: int) -> WorkloadShape:  # batch_size is intentionally ignored
        elements = self.matrix_dim * self.matrix_dim
        return WorkloadShape(elements * 8, elements * 4, self.matrix_dim**3)

    def make_input(self, seed: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        del batch_size
        return (
            _random_tensor(seed, (self.matrix_dim, self.matrix_dim)),
            _random_tensor(seed + 1, (self.matrix_dim, self.matrix_dim)),
        )

    def run(self, value: TensorTree) -> torch.Tensor:
        assert isinstance(value, tuple)
        left, right = value
        return left @ right

    def output_shape(self, batch_size: int) -> tuple[int, ...]:
        del batch_size
        return (self.matrix_dim, self.matrix_dim)

    def items_in_batch(self, batch_size: int) -> int:
        del batch_size
        return 1


@dataclass(frozen=True)
class MultiwordArithmeticWorkload:
    """Generic 64-bit multiword arithmetic over sequential numeric vectors.

    A vector is an integer counter and three derived numeric limbs. This is
    intentionally unrelated to elliptic curves, public/private keys, or any
    cryptographic protocol.
    """

    name: str = "multiword"
    _modulus: int = 2_147_483_647

    def shape_for(self, batch_size: int) -> WorkloadShape:
        # Four int64 input limbs and four int64 result limbs per vector.
        return WorkloadShape(batch_size * 4 * 8, batch_size * 4 * 8, batch_size * 24)

    def make_input(self, seed: int, batch_size: int) -> torch.Tensor:
        # ``seed`` is the first counter in a contiguous deterministic range.
        counter = torch.arange(seed, seed + batch_size, dtype=torch.int64)
        modulus = self._modulus
        limb1 = torch.remainder(counter * 48_271 + 1, modulus)
        limb2 = torch.remainder(counter * 69_621 + 17, modulus)
        limb3 = torch.remainder(counter * 110_351 + 123, modulus)
        return torch.stack((torch.remainder(counter, modulus), limb1, limb2, limb3), dim=1)

    def run(self, value: TensorTree) -> torch.Tensor:
        assert isinstance(value, torch.Tensor)
        a, b, c, d = value.unbind(dim=1)
        modulus = self._modulus
        # Four bounded signed-int64 output limbs. This gives an exact CPU/CUDA
        # oracle without implementing a cryptographic primitive.
        r0 = torch.remainder(a * 48_271 + b * 69_621 + c * 17 + d, modulus)
        r1 = torch.remainder(b * 110_351 + c * 48_271 + d * 73 + a, modulus)
        r2 = torch.remainder(c * 69_621 + d * 110_351 + a * 19 + b, modulus)
        r3 = torch.remainder(d * 48_271 + a * 69_621 + b * 29 + c, modulus)
        return torch.stack((r0, r1, r2, r3), dim=1)

    def output_shape(self, batch_size: int) -> tuple[int, ...]:
        return (batch_size, 4)

    def items_in_batch(self, batch_size: int) -> int:
        return batch_size


def build_workload(name: str, width: int, matrix_dim: int) -> Workload:
    if name == "elementwise":
        return ElementwiseWorkload(width)
    if name == "reduction":
        return ReductionWorkload(width)
    if name == "matmul":
        return MatmulWorkload(matrix_dim)
    if name == "multiword":
        return MultiwordArithmeticWorkload()
    raise ValueError(f"Unknown workload: {name}")


def tree_to_device(value: TensorTree, device: torch.device, non_blocking: bool) -> TensorTree:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=non_blocking)
    return tuple(item.to(device=device, non_blocking=non_blocking) for item in value)


def tree_pin_memory(value: TensorTree) -> TensorTree:
    if isinstance(value, torch.Tensor):
        return value.pin_memory()
    return tuple(item.pin_memory() for item in value)
