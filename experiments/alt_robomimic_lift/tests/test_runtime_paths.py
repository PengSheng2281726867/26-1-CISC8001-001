"""Path portability checks for Windows-authored run manifests consumed in WSL."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from alt_lift.repro import resolve_config_path


class RuntimePathTests(unittest.TestCase):
    def test_relative_windows_separators_resolve_inside_experiment(self):
        resolved = resolve_config_path("artifacts\\runs\\lift30\\memory_bank.npz", base=HERE)
        self.assertEqual(resolved, HERE / "artifacts" / "runs" / "lift30" / "memory_bank.npz")

    def test_windows_absolute_paths_are_understood_in_active_runtime(self):
        resolved = resolve_config_path("D:\\Data\\lift\\image.hdf5", base=HERE)
        if sys.platform.startswith("linux"):
            self.assertEqual(resolved, Path("/mnt/d/Data/lift/image.hdf5"))
        else:
            self.assertEqual(str(resolved).lower(), "d:\\data\\lift\\image.hdf5")


if __name__ == "__main__":
    unittest.main()
