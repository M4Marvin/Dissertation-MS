import numpy as np
from src.Point import Point


def test_point_creation():
    p = Point(np.array([0, 0, 0]))
    assert isinstance(p, Point)
    assert p.dimensions == 3
    assert np.all(p.coordinates == np.array([0, 0, 0]))


def test_point_distance():
    p1 = Point(np.array([0, 0, 0]))
    p2 = Point(np.array([1, 1, 1]))
    assert np.isclose(p1.distance(p2), np.sqrt(3))


def test_point_angle():
    p1 = Point(np.array([1, 0, 0]))
    p2 = Point(np.array([0, 0, 0]))
    p3 = Point(np.array([0, 1, 0]))
    assert np.isclose(p1.angle(p1, p2, p3), np.pi / 2)


def test_point_dihedral():
    p1 = Point(np.array([1, 0, 0]))
    p2 = Point(np.array([0, 0, 0]))
    p3 = Point(np.array([0, 1, 0]))
    p4 = Point(np.array([0, 1, 1]))
    assert np.isclose(p1.dihedral(p1, p2, p3, p4), np.pi / 2)
