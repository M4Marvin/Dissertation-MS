from typing import Union, Tuple

import numpy as np
from Bio.PDB.Chain import Chain as BioChain
from Bio.PDB.Residue import Residue


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
        return None


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
