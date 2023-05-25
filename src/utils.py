from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain as BioChain
from Bio.PDB.Residue import Residue
from Bio.PDB.Polypeptide import is_aa, is_nucleic

from src.Point import Point

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

element_masses = {
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.9994,
    "S": 32.065,
    "P": 30.973762,
    "H": 1.00794,
    "F": 18.9984032,
    "CL": 35.453,
    "BR": 79.904,
    "I": 126.90447,
}

residue_info = {
    "type_1": type_1_residues,
    "type_2": type_2_residues,
    "backbone_atoms": backbone_atoms,
}


nucleotide_names = ["DA", "DT", "DG", "DC"]

phosphate_atoms = ["OP1", "OP2", "P", "O5'"]
sugar_atoms = ["C1'", "C2'", "C3'", "C4'", "C5'", "O3'", "O4'"]


class AtomHelper:
    @staticmethod
    def get_atom_type(residue_type: str, atom: Atom) -> str:
        """
        Returns the type of the atom. The atom can be a backbone, sidechain,
        sugar or phosphate atom.

        Parameters
        ----------
        residue_type : str
            The type of the residue to which the atom belongs.
        atom : Atom
            The atom to be processed.

        Returns
        -------
        str
            The type of the atom.
        """
        atom_name = atom.get_name()

        if residue_type in ["type_1", "type_2"]:
            if atom_name in residue_info["backbone_atoms"]:
                return "backbone"
            else:
                return "sidechain"

        elif residue_type == "ssdna":
            if atom_name in sugar_atoms:
                return "sugar"
            elif atom_name in phosphate_atoms:
                return "phosphate"
            else:
                return "base"

        else:
            raise ValueError("Residue type is not valid.")

    @staticmethod
    def element_to_mass(element: str) -> float:
        """
        Returns the mass of the element.

        Parameters
        ----------
        element : str
            The element to be processed.

        Returns
        -------
        float
            The mass of the element.
        """
        return element_masses[element]


class ChainTypeHelper:
    @staticmethod
    def get_chain_type(chain: BioChain) -> str:
        """
        Returns the type of the chain. The chain can be a protein, ssdna.

        Parameters
        ----------
        chain : BioChain
            The chain to be processed.

        Returns
        -------
        str
            The type of the chain.
        """
        for residue in chain:
            if residue.get_resname() in residue_names:
                return "protein"
            elif residue.get_resname() in nucleotide_names:
                return "ssdna"
        return "unknown"


class UnitTypeHelper:
    @staticmethod
    def get_unit_type(unit: Residue) -> str:
        """
        Returns the type of the unit. The unit can be a type_1 residue, type_2
        residue or ssdna.

        Parameters
        ----------
        unit : Residue
            The unit to be processed.

        Returns
        -------
        str
            The type of the unit.
        """
        if is_aa(unit):
            if unit.get_resname() in residue_info["type_1"]:
                return "type_1"
            elif unit.get_resname() in residue_info["type_2"]:
                return "type_2"
        elif is_nucleic(unit):
            return "ssdna"
        else:
            print(unit.get_resname())
            raise ValueError("Unit is not a protein or ssdna residue.")


class NucleotideAtomHelper:
    @staticmethod
    def get_nucleotide_atom_type(atom) -> str:
        """
        Returns the type of the atom. The atom can be a sugar, base or
        phosphate atom.

        Parameters
        ----------
        atom : Atom
            The atom to be processed.

        Returns
        -------
        str
            The type of the atom.
        """

        return AtomHelper.get_atom_type("ssdna", atom)


class COMHelper:
    @staticmethod
    def _center_of_mass(points: List[Point], element_names: List[str]) -> Point:
        """
        Calculates the center of mass of a set of points.

        Args:
            points (List[Point]): List of Point objects.
            element_names (List[str]): List of element names.

        Returns:
            Point: Center of mass of the set of points.
        """
        masses = [AtomHelper.element_to_mass(element) for element in element_names]
        coordinates = np.average(
            [point.coordinates for point in points], axis=0, weights=masses
        )
        return Point(coordinates)

    @staticmethod
    def get_geometric_center(atoms: List[Atom]) -> Union[Point, None]:
        """
        Calculates the geometric center of a set of atoms.

        Args:
            atoms (List[Atom]): List of Atom objects.

        Returns:
            Point: Geometric center of the set of atoms or None
            if the list of atoms is empty.

        """

        if len(atoms) == 0:
            return None

        points = [Point(atom.get_coord()) for atom in atoms]
        return COMHelper._center_of_mass(points, ["C"] * len(atoms))

    @staticmethod
    def center_of_mass(atoms: List[Atom]) -> Union[Point, None]:
        """
        Calculates the center of mass of a set of atoms.

        Args:
            atoms (List[Atom]): List of Atom objects.

        Returns:
            Point: Center of mass of the set of atoms or None
            if the list of atoms is empty.

        """

        if len(atoms) == 0:
            return None

        element_names = [atom.element for atom in atoms]
        points = [Point(atom.get_coord()) for atom in atoms]
        return COMHelper._center_of_mass(points, element_names)

    @staticmethod
    def get_sidechain_com(residue: Residue) -> Union[np.ndarray, None]:
        """
        Returns the center of mass of the sidechain of the residue.

        Parameters
        ----------
        residue : Residue
            The residue to be processed.

        Returns
        -------
        Union[np.ndarray, None]
            The center of mass of the sidechain of the residue.
            Can be None if the residue does not have a sidechain.
        """

        sidechain_atoms = [
            atom
            for atom in residue.get_atoms()
            if atom.get_name() not in residue_info["backbone_atoms"]
        ]

        return (
            COMHelper.center_of_mass(sidechain_atoms)
            if len(sidechain_atoms) > 0
            else None
        )

    @staticmethod
    def process_protein_residue(
        protein: Residue,
    ) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        """
        Returns the center of mass of the residue and the center of mass
        of the sidechain of the residue.

        Parameters
        ----------
        protein : Residue
            The residue to be processed.

        Returns
        -------
        Tuple[np.ndarray, Union[np.ndarray, None]]
            The center of mass of the residue and the center of mass of the
            sidechain of the residue.
        """
        ca_coords = protein["CA"].get_coord()
        ca_coords = Point(ca_coords)
        sidechain_com = COMHelper.get_sidechain_com(protein)
        return ca_coords, sidechain_com

    @staticmethod
    def get_nucleotide_coms(
        nucleotide: Residue,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the center of mass of the sugar, base and phosphate of the
        nucleotide.

        Parameters
        ----------
        nucleotide : Residue
            The nucleotide to be processed.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            The center of mass of the sugar, base and phosphate of the
            nucleotide.
        """

        sugar_atoms = []
        base_atoms = []
        phosphate_atoms = []

        for atom in nucleotide.get_atoms():
            atom_type = NucleotideAtomHelper.get_nucleotide_atom_type(atom)
            if atom_type == "sugar":
                sugar_atoms.append(atom)
            elif atom_type == "base":
                base_atoms.append(atom)
            elif atom_type == "phosphate":
                phosphate_atoms.append(atom)

        sugar_com = COMHelper.center_of_mass(sugar_atoms)
        base_com = COMHelper.center_of_mass(base_atoms)
        phosphate_com = COMHelper.center_of_mass(phosphate_atoms)

        return sugar_com, base_com, phosphate_com


def combine_dicts(dict_list: List[Dict]) -> Dict:
    # Initialize a new dictionary
    combined_dict: Dict = {}

    for d in dict_list:
        for key, value in d.items():
            # If the key is not yet in the new dict, add it
            if key not in combined_dict:
                combined_dict[key] = value
            else:
                # If the value is a numpy array, use numpy's concatenate
                if isinstance(value, np.ndarray):
                    combined_dict[key] = np.concatenate((combined_dict[key], value))
                # If the value is a pandas DataFrame, use pandas' concat
                elif isinstance(value, pd.DataFrame):
                    combined_dict[key] = pd.concat(
                        [combined_dict[key], value]
                    ).reset_index(drop=True)

    return combined_dict


def print_key_shapes(dictionary: Dict) -> None:
    for key, value in dictionary.items():
        print(f"{key}: {value.shape}")
