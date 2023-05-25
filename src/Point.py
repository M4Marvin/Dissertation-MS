from dataclasses import dataclass, field
from typing import Union

import numpy as np


@dataclass
class Point:
    """
    Class Point to represent a point in a multidimensional space.

    Attributes:
        _coordinates: np.ndarray
            Coordinates of the point in the space.
            This will be used for calculations and binary operations.
    """

    _coordinates: np.ndarray = field(init=True, repr=True, compare=True)

    def __post_init__(self):
        """
        Initializes the coordinates attribute post object instantiation.
        """
        self._coordinates = np.array(self._coordinates)

    @property
    def coordinates(self) -> np.ndarray:
        """
        Returns the coordinates of the point.

        Returns:
            np.ndarray: Coordinates of the point.
        """
        return self._coordinates

    @property
    def dimensions(self) -> int:
        """
        Returns the dimensions of the space.

        Returns:
            int: Number of dimensions of the space.
        """
        return len(self._coordinates)

    def __str__(self) -> str:
        """
        Returns a string representation of the point.

        Returns:
            str: String representation of the point.
        """
        return np.array2string(self._coordinates, precision=3, suppress_small=True)

    def __repr__(self) -> str:
        return self.__str__()


def distance(point_1: Point, point_2: Point) -> Union[np.float64, None]:
    """
    Calculates the distance between two points.

    Args:
        point_1 (Point): First Point object.
        point_2 (Point): Second Point object.

    Returns:
        np.float64: Distance between the two points. If any of the points is
        None, returns None.
    """
    if point_1 is None or point_2 is None:
        return None
    return np.linalg.norm(point_1.coordinates - point_2.coordinates)


def vector(
    point1: Union[Point, None], point2: Union[Point, None]
) -> Union[np.ndarray, None]:
    """
    Calculates the vector from point1 to point2.

    Args:
        point1 (Point): Start point.
        point2 (Point): End point.

    Returns:
        np.ndarray: Vector from point1 to point2. If any of the points is None,
        returns None.
    """
    if point1 is None or point2 is None:
        return None
    return point2.coordinates - point1.coordinates


def angle(
    point1: Union[Point, None],
    point2: Union[Point, None],
    point3: Union[Point, None],
) -> Union[np.float64, None]:
    """
    Calculates the angle (in degrees) created by three points, with the
    second point as the vertex.

    Args:
        point1 (Point): First Point object.
        point2 (Point): Second Point object (vertex of the angle).
        point3 (Point): Third Point object.

    Returns:
        np.float64: Angle in degrees between the three points. If any of the
        points is None, returns None.
    """
    if point1 is None or point2 is None or point3 is None:
        return None
    vector1 = vector(point2, point1)
    vector2 = vector(point2, point3)
    cosine_angle: np.float64 = np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def dihedral(
    point1: Union[Point, None],
    point2: Union[Point, None],
    point3: Union[Point, None],
    point4: Union[Point, None],
) -> Union[np.float64, None]:
    """
    Calculates the dihedral angle (in degrees) created by four points.

    Args:
        point1 (Point): First Point object.
        point2 (Point): Second Point object.
        point3 (Point): Third Point object.
        point4 (Point): Fourth Point object.

    Returns:
        float: Dihedral angle in degrees between the four points.
    """
    if point1 is None or point2 is None or point3 is None or point4 is None:
        return None

    vector1 = vector(point1, point2)
    vector2 = vector(point2, point3)
    vector3 = vector(point3, point4)

    normal1 = np.cross(vector1, vector2)
    normal2 = np.cross(vector2, vector3)

    cosine_angle = np.dot(normal1, normal2)
    cosine_angle /= np.linalg.norm(normal1) * np.linalg.norm(normal2)

    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
