import unittest
import numpy as np
from src import decision_logic
from src import metrics

class TestCoreLogic(unittest.TestCase):
    def test_evaluate_drop_decision(self):
        self.assertEqual(decision_logic.evaluate_drop_decision(0.9, 0.8), "DROP")
        self.assertEqual(decision_logic.evaluate_drop_decision(0.7, 0.8), "NO DROP")
        self.assertEqual(decision_logic.evaluate_drop_decision(0.8, 0.8), "DROP")

    def test_compute_hit_probability(self):
        impact_points = np.array([[0, 0], [1, 0], [2, 0], [10, 0]])
        target_pos = np.array([0, 0])
        target_radius = 2.5
        # Points within radius 2.5: (0,0), (1,0), (2,0). 3 hits.
        # Total points: 4.
        # Probability: 3/4 = 0.75
        prob = metrics.compute_hit_probability(impact_points, target_pos, target_radius)
        self.assertEqual(prob, 0.75)

    def test_compute_cep50(self):
        # Distances: 0, 1, 2, 10
        # Median of [0, 1, 2, 10] is (1+2)/2 = 1.5
        impact_points = np.array([[0, 0], [1, 0], [2, 0], [10, 0]])
        target_pos = np.array([0, 0])
        cep = metrics.compute_cep50(impact_points, target_pos)
        self.assertEqual(cep, 1.5)

if __name__ == "__main__":
    unittest.main()
