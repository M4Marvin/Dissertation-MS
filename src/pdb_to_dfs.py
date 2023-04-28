from Bio.PDB.Residue import Residue
from Bio.PDB.Chain import Chain as BioChain
from Bio.PDB.Structure import Structure as BioStructure
import pandas as pd
import numpy as np
import warnings

from dataclasses import dataclass
from src.utils import residue_names, nucleotide_names, residue_info
from typing import Tuple, Union, List, Dict, Any

warnings.filterwarnings("ignore")

processed_folder = "./data/processed_pdbs"


class ChainTypeHelper:
    @staticmethod
    def get_chain_type(chain: BioChain) -> str:
        for residue in chain:
            if residue.get_resname() in residue_names:
                return "protein"
            elif residue.get_resname() in nucleotide_names:
                return "ssdna"
        return None


class UnitTypeHelper:
    @staticmethod
    def get_unit_type(unit: Residue) -> str:
        if unit.get_resname() in residue_names:
            if unit.get_resname() in residue_info["type_1"]:
                return "type_1"
            elif unit.get_resname() in residue_info["type_2"]:
                return "type_2"
        elif unit.get_resname() in nucleotide_names:
            return "ssdna"
        else:
            raise ValueError("Unit is not a protein or ssdna residue.")


class NucleotideAtomTypeHelper:
    @staticmethod
    def get_nucleotide_atom_type(atom) -> str:
        atom_name = atom.get_name()
        phosphate_atoms = ["P", "OP1", "OP2"]
        if atom_name in phosphate_atoms:
            return "phosphate"
        elif "'" in atom_name:
            return "sugar"
        else:
            return "base"


class COMHelper:
    @staticmethod
    def get_sidechain_com(residue: Residue) -> Union[np.ndarray, None]:
        sidechain_coords = [
            atom.get_coord()
            for atom in residue.get_atoms()
            if atom.get_name() not in ["N", "C", "O", "CA"]
        ]
        return np.mean(sidechain_coords, axis=0) if sidechain_coords else None

    @staticmethod
    def process_protein_residue(
        protein: Residue,
    ) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        ca_coords = next(
            (
                atom.get_coord()
                for atom in protein.get_atoms()
                if atom.get_name() == "CA"
            ),
            None,
        )
        sidechain_com = COMHelper.get_sidechain_com(protein)
        return ca_coords, sidechain_com

    @staticmethod
    def get_nucleotide_coms(
        nucleotide: Residue,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sugar_coords = []
        base_coords = []
        phosphate_coords = []

        for atom in nucleotide.get_atoms():
            atom_type = NucleotideAtomTypeHelper.get_nucleotide_atom_type(atom)
            if atom_type == "sugar":
                sugar_coords.append(atom.get_coord())
            elif atom_type == "base":
                base_coords.append(atom.get_coord())
            elif atom_type == "phosphate":
                phosphate_coords.append(atom.get_coord())

        if sugar_coords == []:
            sugar_com = None
        else:
            sugar_com = np.mean(sugar_coords, axis=0)

        if base_coords == []:
            base_com = None
        else:
            base_com = np.mean(base_coords, axis=0)

        if phosphate_coords == []:
            phosphate_com = None
        else:
            phosphate_com = np.mean(phosphate_coords, axis=0)

        return sugar_com, base_com, phosphate_com


class ChainUnit:
    def __init__(self, unit: Residue):
        self.unit = unit
        self.unit_type = UnitTypeHelper.get_unit_type(self.unit)
        self.fidelity = self.is_fidelity_valid()
        self.resname = self.unit.get_resname()
        self.id = self.unit.get_id()[1]
        self.coms = self.get_coms()

    def __str__(self, include_coms=False) -> str:
        out = ""
        out += f"Name of the unit: {self.resname}\n"
        out += f"ID of the unit: {self.id}\n"
        out += f"Type of the unit: {self.unit_type}\n"
        if include_coms:
            out += f"Center of masses: {self.coms}\n"
        return out

    def is_fidelity_valid(self, print_info=False) -> bool:
        if self.unit_type in ["type_1", "type_2"]:
            return self._check_backbone_atoms(print_info)
        elif self.unit_type == "ssdna":
            return self._check_phosphate_and_sugar_atoms(print_info)
        return False

    def _check_backbone_atoms(self, print_info=False) -> bool:
        backbone_atoms = {"N", "CA", "C", "O"}
        present_atoms = {atom.get_name() for atom in self.unit.get_atoms()}

        if print_info:
            print(f"Backbone atoms: {backbone_atoms}")
            print(f"Present atoms: {present_atoms}")

        return backbone_atoms.issubset(present_atoms)

    def _check_phosphate_and_sugar_atoms(self, print_info=False) -> bool:
        phosphate_atoms = {"P", "OP1", "OP2"}
        sugar_atoms = {"C1'", "C2'", "C3'", "C4'", "C5'", "O4'", "O5'"}
        present_atoms = {atom.get_name() for atom in self.unit.get_atoms()}
        if print_info:
            print(f"Phosphate atoms: {phosphate_atoms}")
            print(f"Sugar atoms: {sugar_atoms}")
            print(f"Present atoms: {present_atoms}")
        return phosphate_atoms.issubset(present_atoms) and sugar_atoms.issubset(
            present_atoms
        )

    def get_coms(self):
        # Check for the fidelity of the unit
        if not self.fidelity:
            return None

        if self.unit_type == "type_1":
            # Get the center of mass of the alpha carbon
            ca_coords = self.unit["CA"].get_coord()
            return ca_coords, None

        elif self.unit_type == "type_2":
            # Get the center of mass of the alpha carbon and the sidechain
            ca_coords = self.unit["CA"].get_coord()
            sidechain_com = COMHelper.get_sidechain_com(self.unit)
            return ca_coords, sidechain_com

        elif self.unit_type == "ssdna":
            # Get the center of mass of the phosphate, sugar, and base
            phosphate_com, sugar_com, base_com = COMHelper.get_nucleotide_coms(
                self.unit
            )
            return phosphate_com, sugar_com, base_com

        else:
            return None

    def to_dict(self) -> Dict[str, Any]:
        if self.fidelity is False:
            return None

        unit_dict = {
            "resname": self.resname,
            "id": self.id,
        }

        if self.unit_type == "type_1" or self.unit_type == "type_2":
            ca_coords, sidechain_com = self.coms
            unit_dict["ca_coords"] = ca_coords.tolist()
            if sidechain_com is not None:
                unit_dict["sidechain_com"] = sidechain_com.tolist()
            else:
                unit_dict["sidechain_com"] = None

        elif self.unit_type == "ssdna":
            phosphate_com, sugar_com, base_com = self.coms
            unit_dict["phosphate_com"] = phosphate_com.tolist()
            unit_dict["sugar_com"] = sugar_com.tolist()
            unit_dict["base_com"] = base_com.tolist()

        return unit_dict


@dataclass
class Chain:
    chain: BioChain
    chain_type: str = None
    units: List[ChainUnit] = None

    def __post_init__(self):
        self.chain_type = ChainTypeHelper.get_chain_type(self.chain)
        self.units = self._generate_units()

    def _generate_units(self) -> List[ChainUnit]:
        return [ChainUnit(unit) for unit in self.chain.get_residues()]

    def __str__(self) -> str:
        out = (
            f"Chain type: "
            f"{self.chain_type}\n"
            f"Number of units: {len(self.units)}\n"
        )
        return out

    def create_dataframe(self) -> pd.DataFrame:
        unit_dicts = [unit.to_dict() for unit in self.units if unit.fidelity]
        df = pd.DataFrame(unit_dicts)
        return df


@dataclass
class Structure:
    structure: BioStructure
    chains: List[Chain] = None

    def __post_init__(self):
        self.chains = self._generate_chains()

    def _generate_chains(self) -> List[BioChain]:
        return [Chain(chain) for chain in self.structure.get_chains()]

    def __str__(self) -> str:
        out = f"Number of chains: {len(self.chains)}\n"
        return out

    def to_dfs(self) -> List[pd.DataFrame]:
        dfs = [chain.to_df() for chain in self.chains]
        chain_types = [chain.chain_type for chain in self.chains]
        return dfs, chain_types
