import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from experiments import manual_execution as ops
from experiments.reporting import guard_cli_output, render_summary, write_summary
from experiments.terminal_menu import TerminalMenu
from src.feature_extractor import FEATURE_GROUPS


class TerminalMenuTests(unittest.TestCase):
    def test_custom_groups_are_listed_dynamically_and_invalid_input_retries(self):
        answers = iter(("4", "99", "8 1"))
        with redirect_stdout(io.StringIO()) as terminal:
            groups = TerminalMenu(lambda _: next(answers)).choose_feature_groups()
        self.assertEqual(groups, (tuple(FEATURE_GROUPS)[0], tuple(FEATURE_GROUPS)[7]))
        for name, columns in FEATURE_GROUPS.items():
            self.assertIn(f"{name} ({len(columns)} features)", terminal.getvalue())
        self.assertIn("Seleção inválida", terminal.getvalue())

    def test_presets_have_expected_sizes_and_custom_selection_can_return(self):
        with redirect_stdout(io.StringIO()):
            for option, expected in (("1", 15), ("2", 21), ("3", 25)):
                groups = TerminalMenu(lambda _: option).choose_feature_groups()
                self.assertEqual(sum(len(FEATURE_GROUPS[name]) for name in groups), expected)
            answers = iter(("4", "0"))
            self.assertIsNone(TerminalMenu(lambda _: next(answers)).choose_feature_groups())

    def test_independent_stages_reuse_existing_datasets_and_model(self):
        with TemporaryDirectory() as temporary, patch.object(ops, "MANUAL", Path(temporary)):
            source = ops.safe_run_path("baseline", "existing")
            source.mkdir(parents=True)
            for name in ("train.npz", "validation.npz", "random_forest.joblib"):
                (source / name).touch()
            answers = iter(("new", "1", "new", "1", "2", "0,75"))
            menu = TerminalMenu(lambda _: next(answers))
            with (patch.object(ops, "BASELINE", Path(temporary) / "baseline-original"),
                  patch.object(ops, "OPTIMIZATION", Path(temporary) / "optimization"),
                  patch.object(ops, "build_baseline_copy") as build,
                  patch.object(ops, "train_baseline_copy") as train,
                  patch.object(ops, "evaluate_saved_model") as evaluate,
                  patch.object(ops, "process", side_effect=lambda title, output, operation, **kw: operation())):
                menu.train()
                menu.evaluate()
            output = ops.safe_run_path("baseline", "new")
            build.assert_not_called()
            train.assert_called_once_with(output, source)
            evaluate.assert_called_once_with(output, source / "random_forest.joblib", None,
                                             threshold=0.75, optimize=False)

    def test_invalid_option_and_back_from_submenus(self):
        answers = iter(("inválida", "6", "1", "0", "0", "7", "0", "0"))
        with redirect_stdout(io.StringIO()) as terminal:
            TerminalMenu(lambda _: next(answers)).run()
        self.assertIn("Opção inválida", terminal.getvalue())
        self.assertIn("D01", terminal.getvalue())
        self.assertIn("E01.1", terminal.getvalue())

    def test_overwrite_requires_explicit_confirmation(self):
        with TemporaryDirectory() as temporary, patch.object(ops, "MANUAL", Path(temporary)):
            root = ops.safe_run_path("baseline", "demo")
            root.mkdir(parents=True)
            (root / "summary.md").write_text("preservar", encoding="utf-8")
            operation = Mock()
            with redirect_stdout(io.StringIO()):
                self.assertFalse(TerminalMenu(lambda _: "sim").execute("Etapa", root, ("summary.md",), operation))
                operation.assert_not_called()
                self.assertTrue(TerminalMenu(lambda _: "SIM").execute("Etapa", root, ("summary.md",), operation))
                operation.assert_called_once()
            with self.assertRaises(ValueError):
                ops.safe_run_path("baseline", "../../region_full")

    def test_validation_of_one_variant_reuses_models_without_rendering(self):
        with TemporaryDirectory() as temporary, patch.object(ops, "MANUAL", Path(temporary)):
            root = ops.safe_run_path("E01_1_multiscale_context", "demo")
            datasets, models = root / "datasets", root / "models"
            answers = iter(("demo", "1"))
            menu = TerminalMenu(lambda _: next(answers))
            with (patch.object(menu, "source", side_effect=(datasets, models)),
                  patch.object(ops, "validate_datasets") as check,
                  patch.object(ops, "require_files"),
                  patch("experiments.terminal_menu.e01.run_evaluation") as evaluate,
                  patch.object(ops, "process", side_effect=lambda title, output, operation, **kw: operation()),
                  patch.object(menu, "train_optimization") as train):
                menu.run_optimization("E01_1_multiscale_context", "validate")
            train.assert_not_called()
            check.assert_called_once_with(datasets, experimental=True)
            self.assertEqual(evaluate.call_args.kwargs, {"variants": ("E01.1-A",), "prepared": datasets,
                "models_root": models, "reuse_models": True, "render": False})

    def test_render_stage_has_no_training_or_validation_call(self):
        with TemporaryDirectory() as temporary, patch.object(ops, "MANUAL", Path(temporary)):
            answers = iter(("demo", "2"))
            with (patch.object(ops, "require_files"),
                  patch.object(ops, "process", side_effect=lambda title, output, operation, **kw: operation()),
                  patch("experiments.terminal_menu.e01.render_saved_evaluation") as render,
                  patch("experiments.terminal_menu.e01.run_evaluation") as evaluate):
                TerminalMenu(lambda _: next(answers)).run_optimization("E01_1_multiscale_context", "render")
            evaluate.assert_not_called()
            self.assertEqual(render.call_args.args[1].name, "E01.1-B")

    def test_summary_is_read_only_until_explicitly_written(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "global_metrics.json"
            path.write_text(json.dumps({"comparison": [{"variant": "A", "policy": "default",
                "IoU": 0.25, "TP": 5, "FP": 10, "mean_iou_with_stain": 0.3}]}), encoding="utf-8")
            self.assertIn("25.00%", render_summary(root))
            self.assertFalse((root / "summary.md").exists())
            self.assertIn("A · default", write_summary(root).read_text(encoding="utf-8"))

    def test_cli_refuses_writes_to_another_diagnostic(self):
        with self.assertRaises(SystemExit):
            guard_cli_output(Path("results/diagnostics/D02_false_positives"), (), owner="D01_slic_oracle")
