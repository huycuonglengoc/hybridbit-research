from __future__ import annotations

import unittest

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for runtime tests")
class RuntimeTests(unittest.TestCase):
    def test_task_graph_releases_dependencies(self) -> None:
        from hetero_pipeline.runtime import Task, TaskGraph

        task0 = Task(0, "w", "op", 4, 4, 0, 1, (), ("cpu",), 1, 1)
        task1 = Task(1, "w", "op", 4, 4, 1, 1, (0,), ("cpu",), 2, 1)
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
