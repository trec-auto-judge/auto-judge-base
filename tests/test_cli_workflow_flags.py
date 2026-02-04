"""Tests for CLI workflow override flags (--set, --nset, --jset, --nugget-judge)."""

import unittest
import tempfile
import os

from trec_auto_judge.workflow import KeyValueType
from trec_auto_judge.workflow import load_workflow


class TestKeyValueType(unittest.TestCase):
    """Tests for KeyValueType parameter parsing."""

    def setUp(self):
        self.kv_type = KeyValueType()

    def test_parse_string(self):
        """Plain string value."""
        key, val = self.kv_type.convert("prompt=minimal", None, None)
        self.assertEqual(key, "prompt")
        self.assertEqual(val, "minimal")
        self.assertIsInstance(val, str)

    def test_parse_int(self):
        """Integer value."""
        key, val = self.kv_type.convert("grade_threshold=4", None, None)
        self.assertEqual(key, "grade_threshold")
        self.assertEqual(val, 4)
        self.assertIsInstance(val, int)

    def test_parse_negative_int(self):
        """Negative integer value."""
        key, val = self.kv_type.convert("offset=-10", None, None)
        self.assertEqual(key, "offset")
        self.assertEqual(val, -10)
        self.assertIsInstance(val, int)

    def test_parse_float(self):
        """Float value."""
        key, val = self.kv_type.convert("threshold=0.75", None, None)
        self.assertEqual(key, "threshold")
        self.assertEqual(val, 0.75)
        self.assertIsInstance(val, float)

    def test_parse_bool_true(self):
        """Boolean true values."""
        for true_val in ["true", "True", "TRUE", "yes", "Yes"]:
            key, val = self.kv_type.convert(f"enabled={true_val}", None, None)
            self.assertEqual(key, "enabled")
            self.assertTrue(val)
            self.assertIsInstance(val, bool)

    def test_parse_bool_false(self):
        """Boolean false values."""
        for false_val in ["false", "False", "FALSE", "no", "No"]:
            key, val = self.kv_type.convert(f"enabled={false_val}", None, None)
            self.assertEqual(key, "enabled")
            self.assertFalse(val)
            self.assertIsInstance(val, bool)

    def test_parse_value_with_equals(self):
        """Value containing equals sign."""
        key, val = self.kv_type.convert("formula=a=b+c", None, None)
        self.assertEqual(key, "formula")
        self.assertEqual(val, "a=b+c")

    def test_parse_empty_value(self):
        """Empty value after equals."""
        key, val = self.kv_type.convert("empty=", None, None)
        self.assertEqual(key, "empty")
        self.assertEqual(val, "")

    def test_missing_equals_fails(self):
        """Missing equals sign should fail."""
        from click.exceptions import BadParameter
        with self.assertRaises(BadParameter) as ctx:
            self.kv_type.convert("no_equals", None, None)
        self.assertIn("key=value", str(ctx.exception))


class TestWorkflowLoading(unittest.TestCase):
    """Tests for workflow.yml loading."""

    def test_load_workflow_parses_all_settings(self):
        """load_workflow correctly parses settings from YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
create_nuggets: true
judge: true
nugget_depends_on_responses: false
judge_uses_nuggets: true
force_recreate_nuggets: false
settings:
  filebase: "my-output"
  custom_setting: 10
nugget_settings:
  prompt: "minimal"
judge_settings:
  grade_threshold: 3
""")
            workflow_path = f.name

        try:
            wf = load_workflow(workflow_path)

            # Verify flags
            self.assertTrue(wf.create_nuggets)
            self.assertTrue(wf.judge)
            self.assertFalse(wf.nugget_depends_on_responses)
            self.assertTrue(wf.judge_uses_nuggets)
            self.assertFalse(wf.force_recreate_nuggets)

            # Verify settings
            self.assertEqual(wf.settings["filebase"], "my-output")
            self.assertEqual(wf.settings["custom_setting"], 10)
            self.assertEqual(wf.nugget_settings["prompt"], "minimal")
            self.assertEqual(wf.judge_settings["grade_threshold"], 3)

        finally:
            os.unlink(workflow_path)


if __name__ == "__main__":
    unittest.main()