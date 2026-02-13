
import sys
import unittest
import os
import matplotlib

# Use Agg backend for non-interactive tests
matplotlib.use('Agg')

# Ensure the root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from product.payloads.geometry_validation import (
    validate_geometry,
    validate_aerodynamics,
)
from product.ui.tabs import payload_library


class TestPayloadLibrary(unittest.TestCase):
    def test_imports(self):
        """Test that all necessary modules can be imported."""
        self.assertIsNotNone(validate_geometry)
        self.assertIsNotNone(validate_aerodynamics)
        self.assertIsNotNone(payload_library.PAYLOAD_LIBRARY)

    def test_geometry_validation(self):
        """Test geometry validation logic."""
        self.assertTrue(validate_geometry("sphere", {"diameter_m": 0.1}))
        with self.assertRaises(ValueError):
            validate_geometry("sphere", {"radius": 0.1})

    def test_aerodynamics_validation(self):
        """Test aerodynamic coefficient limits."""
        # Sphere limits: 0.45 - 0.60
        self.assertTrue(validate_aerodynamics("sphere", 0.5))
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            validate_aerodynamics("sphere", 0.1)

    def test_dynamic_builder(self):
        """Test the dynamic builder logic in PayloadLibraryTab."""
        tab = payload_library.PayloadLibraryTab()

        # 1. Select Identity
        # We need to mock the selection
        # Identify index for a generic item
        tab._sync_state_from_archetype(0)
        self.assertIsNone(tab._state["mass"])  # Should be reset

        # 2. Set Mass
        tab._state["mass"] = 10.0

        # 3. Set Geometry (Sphere)
        tab._state["geometry_type"] = "sphere"
        tab._state["dims"] = {"radius": 0.1}
        tab._update_calculations()  # Trigger logic

        # Verify Area (pi * r^2)
        expected_area = 3.14159 * 0.1 * 0.1
        self.assertAlmostEqual(
            tab._state["calculated_area"], expected_area, places=4
        )

        # Verify Density
        expected_vol = (4 / 3) * 3.14159 * 0.1**3
        expected_density = 10.0 / expected_vol
        self.assertAlmostEqual(
            tab._state["calculated_density"], expected_density, places=1
        )

        # Verify Default Cd suggestion
        self.assertEqual(tab._state["drag_coefficient"], 0.47)

        # Verify Config Output
        cfg = tab.get_payload_config()
        self.assertEqual(cfg["mass"], 10.0)
        self.assertEqual(cfg["geometry"]["type"], "sphere")
        self.assertEqual(cfg["geometry"]["dimensions"]["diameter_m"], 0.2)
        self.assertEqual(cfg["cd_source"], "Literature")
        self.assertAlmostEqual(
            cfg["ballistic_coefficient"],
            10.0 / (0.47 * expected_area),
            places=2,
        )

        # 4. Change to Box
        tab._state["geometry_type"] = "box"
        tab._state["dims"] = {"length": 1, "width": 0.5, "height": 0.5}
        tab._state["drag_coefficient"] = None  # Reset to test suggestion
        tab._update_calculations()

        # Verify Area (Max face: 1*0.5 = 0.5)
        self.assertEqual(tab._state["calculated_area"], 0.5)
        self.assertEqual(tab._state["drag_coefficient"], 1.15)

    def test_persistence(self):
        """Test save/load logic (basic file creation)."""
        import os
        import json

        tab = payload_library.PayloadLibraryTab()
        tab._sync_state_from_archetype(0)
        tab._state["mass"] = 5.0
        tab._state["geometry_type"] = "sphere"
        tab._state["dims"] = {"radius": 0.1}
        tab._update_calculations()

        # Inject custom save path or mocking?
        # The method writes to "custom_payloads.json" in CWD.
        # We can try to run it and clean up.
        fname = "custom_payloads.json"
        if os.path.exists(fname):
            os.remove(fname)

        try:
            tab._save_config()
            self.assertTrue(os.path.exists(fname))
            with open(fname, "r") as f:
                data = json.load(f)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["mass"], 5.0)
        finally:
            if os.path.exists(fname):
                os.remove(fname)

if __name__ == "__main__":
    unittest.main()
