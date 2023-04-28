from typing import List
import numpy as np

residue_names = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLU",
    "GLN",
    "GLY",
    "HIS",
    "HYP",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]

type_1_residues = ["GLY", "ALA", "SER", "THR", "CYS", "VAL", "LEU", "PRO", "ASP", "ASN"]

type_2_residues = ["ILE", "MET", "PHE", "TYR", "TRP", "GLU", "GLN", "HIS", "LYS", "ARG"]

backbone_atoms = ["N", "CA", "C", "O"]

residue_info = {
    "type_1": type_1_residues,
    "type_2": type_2_residues,
    "backbone_atoms": backbone_atoms,
}


nucleotide_names = ["DA", "DT", "DG", "DC"]

phosphate_atoms = ["OP1", "OP2", "P"]
sugar_atoms = ["C1'", "C2'", "C3'", "C4'", "C5'", "O3'", "O4'", "O5'"]


def get_chain_type(chain) -> str:
    first_residue = chain.residue(0).name
    if first_residue in residue_names:
        return "protein"
    elif first_residue in nucleotide_names:
        return "ssDNA"
    else:
        return "other"


def get_residue_names_list(chain) -> List[str]:
    return [x.name for x in chain.residues]


def distance(coord1, coord2):
    return np.linalg.norm(coord1 - coord2)


def angle(coord1, coord2, coord3):
    vector1 = coord1 - coord2
    vector2 = coord3 - coord2
    cos_angle = np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    return np.degrees(angle)


def dihedral(coord1, coord2, coord3, coord4):
    # Calculate the vectors between the points
    b1 = coord2 - coord1
    b2 = coord3 - coord2
    b3 = coord4 - coord3

    # Calculate the normal vectors to the planes defined by the
    # first three points and last three points
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    # Calculate the axis of rotation
    m = np.cross(n1, n2)

    # Calculate the sine and cosine of the dihedral angle
    sin_theta = (
        np.linalg.norm(np.cross(n1, m))
        * np.dot(n1, n2)
        / (np.linalg.norm(n1) * np.linalg.norm(n2))
    )
    cos_theta = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))

    # Calculate the dihedral angle in radians and convert to degrees
    theta = np.arctan2(sin_theta, cos_theta)
    theta_degrees = np.degrees(theta)

    return theta_degrees
