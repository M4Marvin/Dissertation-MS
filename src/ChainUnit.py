from typing import Dict, Any
from Bio.PDB import Residue

from src.utils import COMHelper
from src.utils import UnitTypeHelper
from src.Point import Point


class ChainUnit:
    """
    A class to represent a unit in a protein chain.

    ...

    Attributes
    ----------
    unit : Residue
        a Residue object from BioPython
    unit_type : str
        the type of the unit ('type_1', 'type_2', 'ssdna', etc.)
    fidelity : bool
        a flag indicating if the unit is valid
    resname : str
        the name of the residue
    id : int
        the ID of the residue
    coms : tuple[Point]
        the center of masses of the different parts of the unit

    Methods
    -------
    __str__(self, include_coms=False):
        Returns a string representation of the unit.
    is_fidelity_valid(self, print_info=False):
        Checks if the unit is valid.
    _check_backbone_atoms(self, print_info=False):
        Checks if the backbone atoms are present in the unit.
    _check_phosphate_and_sugar_atoms(self, print_info=False):
        Checks if the phosphate and sugar atoms are present in the unit.
    get_coms(self):
        Calculates the center of masses of the different parts of the unit.
    to_dict(self):
        Returns a dictionary representation of the unit.
    """

    def __init__(self, unit: Residue):
        """
        Constructs all the necessary attributes for the ChainUnit object.

        Parameters
        ----------
            unit : Residue
                a Residue object from BioPython
        """

        self.unit = unit
        self.unit_type = UnitTypeHelper.get_unit_type(self.unit)
        self.fidelity = self.is_fidelity_valid()
        self.resname = self.unit.get_resname()
        self.id = self.unit.get_id()[1]
        self.coms = self.get_coms()

    def __str__(self, include_coms=False) -> str:
        """
        Returns a string representation of the unit.

        Parameters
        ----------
            include_coms : bool, optional
                a flag indicating if the center of masses should be included
                in the string (default is False)

        Returns
        -------
            str
                a string representation of the unit
        """

        out = ""
        out += f"Name of the unit: {self.resname}\n"
        out += f"ID of the unit: {self.id}\n"
        out += f"Type of the unit: {self.unit_type}\n"
        if include_coms:
            out += f"Center of masses: {self.coms}\n"
        return out

    def is_fidelity_valid(self, print_info=False) -> bool:
        """
        Checks if the unit is valid, i.e. if it contains the necessary atoms.

        Parameters
        ----------
            print_info : bool, optional
                a flag indicating if the information should be printed
                (default is False)

        Returns
        -------
            bool
                True if the unit is valid, False otherwise
        """
        if self.unit_type in ["type_1", "type_2"]:
            return self._check_backbone_atoms(print_info)
        elif self.unit_type == "ssdna":
            return self._check_phosphate_and_sugar_atoms(print_info)
        return False

    def _check_backbone_atoms(self, print_info=False) -> bool:
        """
        Checks if all the backbone atoms are present in the unit.

        Parameters
        ----------
            print_info : bool, optional
                a flag indicating if the information should be printed
                (default is False)

        Returns
        -------
            bool
                True if the backbone atoms are present, False otherwise
        """

        backbone_atoms = {"N", "CA", "C", "O"}
        present_atoms = {atom.get_name() for atom in self.unit.get_atoms()}

        if print_info:
            print(f"Backbone atoms: {backbone_atoms}")
            print(f"Present atoms: {present_atoms}")

        return backbone_atoms.issubset(present_atoms)

    def _check_phosphate_and_sugar_atoms(self, print_info=False) -> bool:
        """
        Checks if the phosphate and sugar atoms are present in the unit.

        Parameters
        ----------
            print_info : bool, optional
                a flag indicating if the information should be printed
                (default is False)

        Returns
        -------
            bool
                True if the phosphate and sugar atoms are present, False
                otherwise
        """

        phosphate_atoms = {"P", "OP1", "OP2"}
        sugar_atoms = {"C1'", "C2'", "C3'", "C4'", "C5'", "O4'", "O5'"}
        present_atoms = {atom.get_name() for atom in self.unit.get_atoms()}
        if print_info:
            print(f"Phosphate atoms: {phosphate_atoms}")
            print(f"Sugar atoms: {sugar_atoms}")
            print(f"Present atoms: {present_atoms}")
        phosphate_present = phosphate_atoms.issubset(present_atoms)
        sugar_present = sugar_atoms.issubset(present_atoms)
        return phosphate_present and sugar_present

    def get_coms(self):
        """
        Calculates the center of masses of the different parts of the unit.
        The ouptut can be different based on the type of the unit.

        Returns
        -------
            Union[dict, None]
                a dictionary containing the center of masses of the different
                parts
        """

        # Check for the fidelity of the unit
        if not self.fidelity:
            return None

        if self.unit_type == "type_1":
            # Get the center of mass of the alpha carbon
            ca_coords = self.unit["CA"].get_coord()
            ca_coords = Point(ca_coords)
            return {"ca_coords": ca_coords, "sidechain_com": None}

        elif self.unit_type == "type_2":
            # Get the center of mass of the alpha carbon and the sidechain
            ca_coords = self.unit["CA"].get_coord()
            ca_coords = Point(ca_coords)
            sidechain_com = COMHelper.get_sidechain_com(self.unit)
            return {"ca_coords": ca_coords, "sidechain_com": sidechain_com}

        elif self.unit_type == "ssdna":
            # Get the center of mass of the phosphate, sugar, and base
            phosphate_com, sugar_com, base_com = COMHelper.get_nucleotide_coms(
                self.unit
            )
            return {
                "phosphate_com": phosphate_com,
                "sugar_com": sugar_com,
                "base_com": base_com,
            }

        else:
            return None

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the unit.

        Returns
        -------
            dict
                a dictionary representation of the unit
        """
        if self.fidelity is False:
            return None

        unit_dict = {
            "resname": self.resname,
            "id": self.id,
        }

        if self.unit_type == "type_1" or self.unit_type == "type_2":
            unit_dict["unit_type"] = self.unit_type

        elif self.unit_type == "ssdna":
            phosphate_com, sugar_com, base_com = self.coms

        # Merge the dictionaries
        unit_dict = {**unit_dict, **self.coms}

        return unit_dict
