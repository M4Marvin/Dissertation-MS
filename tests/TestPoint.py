import sys

sys.path.append("src")

import unittest
import numpy as np

from typing import Union
from Point import Point, distance, vector, angle, dihedral


class TestPoint(unittest.TestCase):
    def setUp(self):
        self.p1 = Point([1, 2, 3])
        self.p2 = Point([4, 5, 6])
        self.p3 = Point([0, 0, 0])
        self.p4 = Point([1, 0, 0])

    def test_get_coordinates(self):
        self.assertTrue(isinstance(self.p1.get_coordinates(), np.ndarray))
        self.assertEqual(self.p1.get_coordinates().tolist(), [1, 2, 3])

    def test_distance(self):
        self.assertEqual(distance(self.p1, self.p2), np.sqrt(27))
        self.assertIsNone(distance(None, self.p1))
        self.assertIsNone(distance(self.p1, None))
        self.assertIsNone(distance(None, None))

    def test_vector(self):
        self.assertTrue(isinstance(vector(self.p1, self.p2), np.ndarray))
        self.assertEqual(vector(self.p1, self.p2).tolist(), [3, 3, 3])
        self.assertIsNone(vector(None, self.p1))
        self.assertIsNone(vector(self.p1, None))
        self.assertIsNone(vector(None, None))

    def test_angle(self):
        self.assertAlmostEqual(angle(self.p1, self.p2, self.p3), 9.2744998)
        self.assertIsNone(angle(None, self.p2, self.p3))
        self.assertIsNone(angle(self.p1, None, self.p3))
        self.assertIsNone(angle(self.p1, self.p2, None))
        self.assertIsNone(angle(None, None, None))

    def test_dihedral(self):
        self.assertAlmostEqual(dihedral(self.p1, self.p2, self.p3, self.p1), 0.000)
        self.assertIsNone(dihedral(None, self.p2, self.p3, self.p1))
        self.assertIsNone(dihedral(self.p1, None, self.p3, self.p4))
        self.assertIsNone(dihedral(self.p1, self.p2, None, self.p4))
        self.assertIsNone(dihedral(None, None, None, None))


if __name__ == "__main__":
    unittest.main()
