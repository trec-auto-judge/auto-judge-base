import tempfile
import unittest
from pathlib import Path

import yaml

from trec_auto_judge.workflow import Workflow, load_workflow, resolve_config_file_path
from trec_auto_judge.judge_runner import _write_run_config


class TestWorkflowModel(unittest.TestCase):
    """Test Workflow model parsing and defaults."""

    def test_backwards_compatibility_minimal(self):
        """Minimal workflow with just create_nuggets and judge should parse."""
        wf = Workflow(create_nuggets=True, judge=True)

        self.assertTrue(wf.create_nuggets)
        self.assertTrue(wf.judge)
        # New fields should have defaults
        self.assertEqual(wf.settings, {})
        self.assertEqual(wf.nugget_settings, {})
        self.assertEqual(wf.judge_settings, {})
        self.assertEqual(wf.variants, {})
        self.assertEqual(wf.sweeps, {})
        self.assertTrue(wf.nugget_depends_on_responses)
        self.assertTrue(wf.judge_uses_nuggets)
        self.assertIsNone(wf.force_recreate_nuggets)  # None = defaults to create_nuggets at runtime

    def test_settings_fields_parse(self):
        """Settings dicts should parse correctly."""
        wf = Workflow(
            create_nuggets=True,
            judge=True,
            settings={"top_k": 20, "filebase": "{_name}"},
            nugget_settings={"extraction_style": "thorough"},
            judge_settings={"threshold": 0.5},
        )

        self.assertEqual(wf.settings["top_k"], 20)
        self.assertEqual(wf.settings["filebase"], "{_name}")
        self.assertEqual(wf.nugget_settings["extraction_style"], "thorough")
        self.assertEqual(wf.judge_settings["threshold"], 0.5)

    def test_variants_parse(self):
        """Variants dict should parse correctly."""
        wf = Workflow(
            create_nuggets=True,
            judge=True,
            variants={
                "ans-r": {"prompt": "AnswerR", "threshold": 0.8},
                "strict": {"threshold": 0.9},
            },
        )

        self.assertEqual(len(wf.variants), 2)
        self.assertEqual(wf.variants["ans-r"]["prompt"], "AnswerR")
        self.assertEqual(wf.variants["strict"]["threshold"], 0.9)

    def test_sweeps_parse(self):
        """Sweeps with list values should parse correctly."""
        wf = Workflow(
            create_nuggets=True,
            judge=True,
            sweeps={
                "top-k-sweep": {"top_k": [10, 20, 50]},
                "threshold-grid": {"top_k": [10, 20], "threshold": [0.3, 0.5]},
            },
        )

        self.assertEqual(len(wf.sweeps), 2)
        self.assertEqual(wf.sweeps["top-k-sweep"]["top_k"], [10, 20, 50])
        self.assertEqual(wf.sweeps["threshold-grid"]["threshold"], [0.3, 0.5])

    def test_lifecycle_flags(self):
        """Lifecycle flags should parse and override defaults."""
        wf = Workflow(
            create_nuggets=True,
            judge=True,
            nugget_depends_on_responses=False,
            judge_uses_nuggets=False,
            force_recreate_nuggets=True,
        )

        self.assertFalse(wf.nugget_depends_on_responses)
        self.assertFalse(wf.judge_uses_nuggets)
        self.assertTrue(wf.force_recreate_nuggets)


class TestLoadWorkflowFiles(unittest.TestCase):
    """Test loading actual workflow.yml files."""

    def test_load_existing_workflow_files(self):
        """All existing workflow.yml files should load without error."""
        judges_dir = Path(__file__).parent.parent / "trec25" / "judges"

        if not judges_dir.exists():
            self.skipTest("trec25/judges directory not found")

        workflow_files = list(judges_dir.glob("*/workflow.yml"))
        self.assertGreater(len(workflow_files), 0, "No workflow.yml files found")

        for wf_path in workflow_files:
            with self.subTest(workflow=wf_path.parent.name):
                wf = load_workflow(wf_path)
                # Should parse without error and have valid boolean flags
                self.assertIsInstance(wf.create_nuggets, bool)
                self.assertIsInstance(wf.judge, bool)


class TestWriteRunConfig(unittest.TestCase):
    """Test _write_run_config() for reproducibility tracking."""

    def test_resolve_config_file_path_adds_extension(self):
        """Filebase without extension gets .config.yml added."""
        self.assertEqual(
            resolve_config_file_path(Path("rubric")),
            Path("rubric.config.yml"),
        )
        self.assertEqual(
            resolve_config_file_path(Path("output/rubric")),
            Path("output/rubric.config.yml"),
        )

    def test_resolve_config_file_path_preserves_yml(self):
        """Filebase with .yml/.yaml extension is used as-is."""
        self.assertEqual(
            resolve_config_file_path(Path("custom.yml")),
            Path("custom.yml"),
        )
        self.assertEqual(
            resolve_config_file_path(Path("custom.yaml")),
            Path("custom.yaml"),
        )

    def test_writes_required_fields(self):
        """Config file includes all required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test"

            _write_run_config(
                judge_output_path=output_path,
                config_name="my-variant",
                do_create_nuggets=True,
                do_create_qrels=False,
                do_judge=True,
                llm_model="gpt-4o",
                settings=None,
                nugget_settings=None,
                judge_settings=None,
                qrels_settings=None,
            )

            config_path = Path(tmpdir) / "test.config.yml"
            self.assertTrue(config_path.exists())

            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Required fields
            self.assertEqual(config["name"], "my-variant")
            self.assertEqual(config["create_nuggets"], True)
            self.assertEqual(config["create_qrels"], False)
            self.assertEqual(config["judge"], True)
            self.assertEqual(config["llm_model"], "gpt-4o")
            self.assertIn("timestamp", config)
            self.assertIn("git", config)
            self.assertIn("commit", config["git"])
            self.assertIn("dirty", config["git"])
            self.assertIn("remote", config["git"])

    def test_omits_empty_settings(self):
        """Empty settings dicts are not included in config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test"

            _write_run_config(
                judge_output_path=output_path,
                config_name="default",
                do_create_nuggets=False,
                do_create_qrels=False,
                do_judge=True,
                llm_model="llama-3",
                settings=None,
                nugget_settings=None,
                judge_settings=None,
                qrels_settings=None,
            )

            config_path = Path(tmpdir) / "test.config.yml"
            with open(config_path) as f:
                config = yaml.safe_load(f)

            self.assertNotIn("settings", config)
            self.assertNotIn("nugget_settings", config)
            self.assertNotIn("judge_settings", config)
            self.assertNotIn("qrels_settings", config)

    def test_includes_nonempty_settings(self):
        """Non-empty settings dicts are included in config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test"

            _write_run_config(
                judge_output_path=output_path,
                config_name="default",
                do_create_nuggets=True,
                do_create_qrels=True,
                do_judge=True,
                llm_model="gpt-4o",
                settings={"top_k": 20, "filebase": "rubric"},
                nugget_settings={"extraction_style": "thorough"},
                judge_settings={"threshold": 0.5},
                qrels_settings={"grade_range": [0, 3]},
            )

            config_path = Path(tmpdir) / "test.config.yml"
            with open(config_path) as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["settings"]["top_k"], 20)
            self.assertEqual(config["settings"]["filebase"], "rubric")
            self.assertEqual(config["nugget_settings"]["extraction_style"], "thorough")
            self.assertEqual(config["judge_settings"]["threshold"], 0.5)
            self.assertEqual(config["qrels_settings"]["grade_range"], [0, 3])