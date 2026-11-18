from __future__ import annotations

import unittest

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for runtime tests")
class RuntimeTests(unittest.TestCase):
    def test_task_graph_releases_dependencies(self) -> None:
        from hetero_pipeline.runtime import Task, TaskGraph

        task0 = Task(0, "w", "op", 4, 4, 0, 1, (), ("cpu",), 1, 1, 1, 1)
        task1 = Task(1, "w", "op", 4, 4, 1, 1, (0,), ("cpu",), 2, 1, 1, 2)
        graph = TaskGraph([task0, task1])
        self.assertEqual([task.task_id for task in graph.ready(set(), set())], [0])
        self.assertEqual([task.task_id for task in graph.ready({0}, {0})], [1])

    def test_cpu_benchmark_is_correct(self) -> None:
        from hetero_pipeline.runtime import BenchmarkConfig, BenchmarkRunner
        from hetero_pipeline.workloads import ElementwiseWorkload

        config = BenchmarkConfig(
            workload=ElementwiseWorkload(width=8),
            task_count=3,
            batch_size=16,
            cpu_workers=1,
            cpu_threads=1,
            validate_tasks=3,
        )
        result = BenchmarkRunner(config).run("cpu")
        self.assertEqual(len(result.tasks), 3)
        self.assertFalse(result.validation_errors)
        self.assertTrue(result.summary()["validation_passed"])

    def test_multiword_uses_an_exact_counter_range(self) -> None:
        from hetero_pipeline.runtime import BenchmarkConfig, BenchmarkRunner
        from hetero_pipeline.workloads import MultiwordArithmeticWorkload

        runner = BenchmarkRunner(
            BenchmarkConfig(
                workload=MultiwordArithmeticWorkload(),
                total_items=1_000_000,
                batch_size=8192,
                counter_start=1,
            )
        )
        tasks = runner.make_tasks()
        self.assertEqual(sum(task.item_count for task in tasks), 1_000_000)
        self.assertEqual(tasks[0].source_start, 1)
        self.assertEqual(tasks[-1].source_start + tasks[-1].item_count - 1, 1_000_000)

    def test_multiword_output_is_int64(self) -> None:
        from hetero_pipeline.workloads import MultiwordArithmeticWorkload

        workload = MultiwordArithmeticWorkload()
        output = workload.run(workload.make_input(1, 8))
        self.assertEqual(workload.output_dtype(), torch.int64)
        self.assertEqual(output.dtype, torch.int64)

    @unittest.skipUnless(TORCH_AVAILABLE and torch.cuda.is_available(), "CUDA is required for GPU dtype regression test")
    def test_gpu_multiword_preserves_integer_output_dtype(self) -> None:
        from hetero_pipeline.runtime import BenchmarkConfig, BenchmarkRunner
        from hetero_pipeline.workloads import MultiwordArithmeticWorkload

        result = BenchmarkRunner(
            BenchmarkConfig(
                workload=MultiwordArithmeticWorkload(),
                total_items=64,
                batch_size=64,
                validate_tasks=1,
            )
        ).run("gpu")
        self.assertFalse(result.validation_errors)
        self.assertEqual(result.tasks[0].output.dtype, torch.int64)
