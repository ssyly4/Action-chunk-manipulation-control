from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from nero_vla.task_config import emit_shell, load_task


ROOT = Path(__file__).resolve().parents[1]


class TaskConfigTest(unittest.TestCase):
    def test_all_task_presets_are_valid(self) -> None:
        tasks = sorted((ROOT / "config/tasks").glob("*.toml"))
        self.assertEqual([path.stem for path in tasks], ["bottle_to_box_right", "towel_fold"])
        for path in tasks:
            config = load_task(path)
            self.assertTrue(config["arguments"]["values"])

    def test_environment_override_wins_over_default(self) -> None:
        config = load_task(ROOT / "config/tasks/towel_fold.toml")
        with patch.dict(os.environ, {"NERO_POLICY_DURATION": "99"}, clear=False):
            shell = emit_shell(config)
        self.assertNotIn("export NERO_POLICY_DURATION=30", shell)
        self.assertIn("declare -a NERO_TASK_ARGS=", shell)


if __name__ == "__main__":
    unittest.main()
