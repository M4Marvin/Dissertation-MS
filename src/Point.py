from dataclasses import dataclass
import numpy as np


@dataclass
class Point:
    """
    Class Point to represent a point in a multidimensional space.

    Attributes:
        coordinates: np.ndarray
            Coordinates of the point in the space.
        dimensions: int
            Number of dimensions of the space.
    """

    coordinates: np.ndarray
    dimensions: int = None

    def __post_init__(self):
        """
        Initializes the dimensions attribute post object instantiation.
        """
        self.coordinates = np.array(self.coordinates)
        self.dimensions = len(self.coordinates)

    def __str__(self) -> str:
        """
        Returns a string representation of the point.

        Returns:
            str: String representation of the point.
        """
        np.set_printoptions(
            formatter={"float": lambda x: "{0:0.3f}".format(x)},
            linewidth=100,
            suppress=True,
        )
        return str(self.coordinates)

    def __repr__(self) -> str:
        return self.__str__()

    def get_coordinates(self) -> np.ndarray:
        """
        Returns the coordinates of the point.

        Returns:
            np.ndarray: Coordinates of the point.
        """
        return self.coordinates

    def distance(self, other_point) -> float:
        """
        Calculates the Euclidean distance between this point and another point.

        Args:
            other_point (Point): Another Point object.

        Returns:
            float: Euclidean distance between the two points.
        """
        return np.linalg.norm(self.coordinates - other_point.coordinates)

    @classmethod
    def vector(cls, point1, point2):
        """
        Calculates the vector from point1 to point2.

        Args:
            point1 (Point): Start point.
            point2 (Point): End point.

        Returns:
            np.ndarray: Vector from point1 to point2.
        """
        return point2.coordinates - point1.coordinates

    @classmethod
    def angle(cls, point1, point2, point3) -> float:
        """
        Calculates the angle (in radians) created by three points, with the
        second point as the vertex.

        Args:
            point1 (Point): First Point object.
            point2 (Point): Second Point object (vertex of the angle).
            point3 (Point): Third Point object.

        Returns:
            float: Angle in radians between the three points.
        """
        vector1 = cls.vector(point2, point1)
        vector2 = cls.vector(point2, point3)
        cosine_angle = np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2)
        )
        return np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    @classmethod
    def dihedral(cls, point1, point2, point3, point4) -> float:
        """
        Calculates the dihedral angle (in radians) created by four points.

        Args:
            point1 (Point): First Point object.
            point2 (Point): Second Point object.
            point3 (Point): Third Point object.
            point4 (Point): Fourth Point object.

        Returns:
            float: Dihedral angle in radians between the four points.
        """
        vector1 = cls.vector(point1, point2)
        vector2 = cls.vector(point2, point3)
        vector3 = cls.vector(point3, point4)

        normal1 = np.cross(vector1, vector2)
        normal2 = np.cross(vector2, vector3)

        cosine_angle = np.dot(normal1, normal2)
        cosine_angle /= np.linalg.norm(normal1) * np.linalg.norm(normal2)

        return np.arccos(np.clip(cosine_angle, -1.0, 1.0))