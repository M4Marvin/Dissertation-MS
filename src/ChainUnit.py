from typing import Any, Dict, Union

from Bio.PDB import Residue

from src.Point import Point
from src.utils import COMHelper, UnitTypeHelper

backbone_atoms = {"N", "CA", "C", "O"}


class ChainUnit:
    def __init__(self, unit):
        """
        A base class to represent a unit in a protein or nucleotide chain.

        Attributes:
        - unit (Residue): A Residue object from Biopython
        - unit_type (str): The type of the unit ('type_1', 'type_2', 'ssdna', etc.)
        - fidelity (bool): A flag indicating if the unit is valid
        - resname (str): The name of the residue
        - id (int): The ID of the residue
        - coms (tuple[Point]): The center of masses of the different parts of the unit
        """
        self.unit = unit
        self.unit_type = UnitTypeHelper.get_unit_type(unit)
        self.fidelity = self.is_fidelity_valid()
        self.resname = unit.get_resname()
        self.id = unit.get_id()[1]
        self.coms = self.get_coms()

    def __str__(self) -> str:
        """
        Return a string representation of the unit.
        """
        return (
            f"Name of the unit: {self.resname}\n"
            f"ID of the unit: {self.id}\n"
            f"Type of the unit: {self.unit_type}\n"
            f"Center of masses: {self.coms}\n"
        )

    def __repr__(self):
        return self.__str__()

    def is_fidelity_valid(self):
        """
        Check if the unit is valid.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def get_coms(self):
        """
        Calculate the center of masses of the different parts of the unit.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def to_dict(self):
        """
        Return a dictionary representation of the unit.
        """
        if not self.fidelity:
            return None

        unit_dict = {
            "resname": self.resname,
            "id": self.id,
        }

        if self.unit_type in ("type_1", "type_2"):
            unit_dict["type"] = self.unit_type

        unit_dict = {**unit_dict, **self.coms}

        return unit_dict


class ResidueUnit(ChainUnit):
    def is_fidelity_valid(self, print_info=False) -> bool:
        return self._check_backbone_atoms(print_info)

    def _check_backbone_atoms(self, print_info=False) -> bool:
        present_atoms = {atom.get_name() for atom in self.unit.get_atoms()}

        if print_info:
            print(f"Backbone atoms: {backbone_atoms}")
            print(f"Present atoms: {present_atoms}")

        return backbone_atoms.issubset(present_atoms)

    def get_coms(self):
        # Check if the unit is valid
        if not self.fidelity:
            return None

        # Get the coordinates of the CA atom
        ca_coords = self.unit["CA"].get_coord()
        ca_coords = Point(ca_coords)

        # Get the coordinates of the sidechain center of mass,
        # None if the unit is type_1
        sidechain_com = (
            COMHelper.get_sidechain_com(self.unit)
            if self.unit_type == "type_2"
            else None
        )

        # Calculate the geometric center of the unit
        # For type_1 units, the geometric center is the center of mass of the unit
        # For type_2 units, the geometric center is the center of mass of the backbone atoms
        if self.unit_type == "type_1":
            geo_center = COMHelper.get_geometric_center(
                [atom for atom in self.unit.get_atoms()]
            )
        elif self.unit_type == "type_2":
            geo_center = COMHelper.get_geometric_center(
                [
                    atom
                    for atom in self.unit.get_atoms()
                    if atom.get_name() in backbone_atoms
                ]
            )

        return {
            "ca_coords": ca_coords,
            "sidechain_com": sidechain_com,
            "geo_center": geo_center,
        }


class NucleotideUnit(ChainUnit):
    def is_fidelity_valid(self, print_info=False) -> bool:
        return self._check_phosphate_and_sugar_atoms(print_info)

    def _check_phosphate_and_sugar_atoms(self, print_info=False) -> bool:
        phosphate_atoms = {"P", "OP1", "OP2", "O5'"}
        sugar_atoms = {"C1'", "C2'", "C3'", "C4'", "C5'", "O4'"}
        present_atoms = {atom.get_name() for atom in self.unit.get_atoms()}

        if print_info:
            print(f"Phosphate atoms: {phosphate_atoms}")
            print(f"Sugar atoms: {sugar_atoms}")
            print(f"Present atoms: {present_atoms}")

        phosphate_present = phosphate_atoms.issubset(present_atoms)
        sugar_present = sugar_atoms.issubset(present_atoms)
        return phosphate_present and sugar_present

    def get_coms(self):
        if not self.fidelity:
            return None

        phosphate_com, sugar_com, base_com = COMHelper.get_nucleotide_coms(self.unit)
        return {
            "phosphate_com": phosphate_com,
            "sugar_com": sugar_com,
            "base_com": base_com,
        }


class ChainUnitGenerator:
    """
    A class to generate a ResidueUnit or NucleotideUnit object based on the unit type.

    ...

    Attributes
    ----------
    unit : Residue
        a Residue object from BioPython

    Methods
    -------
    generate(self):
        Returns a ResidueUnit or NucleotideUnit object.
    """

    def generate(self, unit: Residue) -> Union[ResidueUnit, NucleotideUnit]:
        """
        Returns a ResidueUnit or NucleotideUnit object based on the unit type.

        Returns
        -------
            Union[ResidueUnit, NucleotideUnit]
                a ResidueUnit or NucleotideUnit object
        """
        unit_type = UnitTypeHelper.get_unit_type(unit)
        if unit_type in ["type_1", "type_2"]:
            out_unit = ResidueUnit(unit)
        elif unit_type == "ssdna":
            out_unit = NucleotideUnit(unit)

        # Check if the unit is valid
        if out_unit.fidelity is False:
            return None

        return out_unit
