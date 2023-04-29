import pandas as pd
import numpy as np
from typing import List
from Bio.PDB.Chain import Chain as BioChain
from src.ChainUnit import ChainUnit
from src.Point import distance, angle, dihedral
from src.utils import ChainTypeHelper
from dataclasses import dataclass


@dataclass
class Chain:
    """
    Class Chain to represent a chain in a biopolymer structure from a PDB file.

    Attributes:
        chain : BioChain
            A BioPython chain object
        chain_type : str, optional
            Type of the chain (e.g. protein, DNA, etc.)
        units : List[ChainUnit], optional
            List of ChainUnit objects representing the residues in the chain
        df : pd.DataFrame, optional
            DataFrame representing the chain
    """

    chain: BioChain
    chain_type: str = None
    units: List[ChainUnit] = None
    df: pd.DataFrame = None

    def __post_init__(self):
        """
        Initializes the chain_type, units and df attributes post object
        instantiation.
        """
        self.chain_type = ChainTypeHelper.get_chain_type(self.chain)
        self.units = self._generate_chain_units()
        self.df = self._construct_dataframe_from_units()

    def _generate_chain_units(self) -> List[ChainUnit]:
        """
        Generates ChainUnit objects from the residues in the BioPython chain.

        Returns:
            List of ChainUnit objects representing each residue in the chain.
        """
        return [ChainUnit(unit) for unit in self.chain.get_residues()]

    def __str__(self) -> str:
        """
        Returns a string representation of the Chain object.

        Returns:
            String representation of the Chain object.
        """
        chain_info = (
            f"Chain type: "
            f"{self.chain_type}\n"
            f"Number of units: {len(self.units)}\n"
        )
        return chain_info

    def _construct_dataframe_from_units(self) -> pd.DataFrame:
        """
        Constructs a DataFrame from the ChainUnit objects in the chain.

        Returns:
            DataFrame representing the chain.
        """
        # Only consider units that have passed the fidelity check
        unit_dicts = [unit.to_dict() for unit in self.units if unit.fidelity]
        df = pd.DataFrame(unit_dicts)
        return df

    def get_bb_distances(self) -> np.ndarray:
        """
        Calculates the distances between the backbone atoms of the units in the
        chain.

        Returns:
            Numpy array of distances between the backbone atoms of the units in
            the chain.
        """
        # The chain should be a protein chain
        assert self.chain_type == "protein"
        bb_distances = []
        for i in range(len(self.units) - 1):
            ca_1 = self.units[i].coms["ca_coords"]
            ca_2 = self.units[i + 1].coms["ca_coords"]
            bb_distances.append(distance(ca_1, ca_2))
        return np.array(bb_distances)

    def get_bs_distances(self) -> pd.DataFrame:
        """
        Calculates the distances between the ca atom and the sidechain
        center of mass of the units in the chain.

        Returns:
            Numpy array of distances between the ca atom and the backbone
            center of mass of the units in the chain.
        """

        # The chain should be a protein chain
        assert self.chain_type == "protein"
        bs_data = {
            "resname": [],
            "distance": [],
        }
        for unit in self.units:
            if unit.fidelity and unit.unit_type == "type_2":
                ca = unit.coms["ca_coords"]
                sc_com = unit.coms["sidechain_com"]
                bs_data["distance"].append(distance(ca, sc_com))
                bs_data["resname"].append(unit.resname)

        return pd.DataFrame(bs_data)

    def get_bbb_angles(self):
        """
        Calculates the angles between the backbone atoms of the units in the
        chain.

        Returns:
            Numpy array of angles between the backbone atoms of the units in
            the chain.
        """
        # The chain should be a protein chain
        assert self.chain_type == "protein"
        bbb_angles = []
        for i in range(len(self.units) - 2):
            ca_1 = self.units[i].coms["ca_coords"]
            ca_2 = self.units[i + 1].coms["ca_coords"]
            ca_3 = self.units[i + 2].coms["ca_coords"]
            bbb_angles.append(angle(ca_1, ca_2, ca_3))
        return np.array(bbb_angles)

    def get_sbb_angles(self):
        """
        Calculates the angles between the sidechain center of mass and the
        backbone atoms of the units in the chain.

        Returns:
            Numpy array of angles between the sidechain center of mass and the
            backbone atoms of the units in the chain.
        """
        # The chain should be a protein chain
        assert self.chain_type == "protein"
        sbb_angles = {
            "resname": [],
            "angle": [],
        }

        for i in range(len(self.units) - 1):
            if self.units[i].fidelity and self.units[i].unit_type == "type_2":
                ca_1 = self.units[i].coms["ca_coords"]
                sc_com = self.units[i].coms["sidechain_com"]
                ca_2 = self.units[i + 1].coms["ca_coords"]
                sbb_angles["angle"].append(angle(sc_com, ca_1, ca_2))
                sbb_angles["resname"].append(self.units[i].resname)

        return pd.DataFrame(sbb_angles)

    def get_bbs_angles(self):
        """
        Calculates the angles between the backbone atoms of the units in the
        chain.

        Returns:
            Numpy array of angles between the backbone atoms of the units in
            the chain.
        """
        # The chain should be a protein chain
        assert self.chain_type == "protein"
        bbs_angles = {
            "resname": [],
            "angle": [],
        }

        for i in range(len(self.units) - 1):
            if self.units[i].fidelity and self.units[i + 1].unit_type == "type_2":
                ca_1 = self.units[i].coms["ca_coords"]
                sc_com = self.units[i + 1].coms["sidechain_com"]
                ca_2 = self.units[i + 1].coms["ca_coords"]
                bbs_angles["angle"].append(angle(ca_1, ca_2, sc_com))
                bbs_angles["resname"].append(self.units[i].resname)

        return pd.DataFrame(bbs_angles)

    def get_bbbb_dihedrals(self):
        """
        Calculates the bbbb dihedral angles.

        Returns:
            Numpy array of dihedral angles between the backbone atoms of
            the units in the chain.
        """
        # The chain should be a protein chain
        assert self.chain_type == "protein"
        bbbb_dihedrals = []
        for i in range(len(self.units) - 3):
            ca_1 = self.units[i].coms["ca_coords"]
            ca_2 = self.units[i + 1].coms["ca_coords"]
            ca_3 = self.units[i + 2].coms["ca_coords"]
            ca_4 = self.units[i + 3].coms["ca_coords"]
            bbbb_dihedrals.append(dihedral(ca_1, ca_2, ca_3, ca_4))
        return np.array(bbbb_dihedrals)

    def get_sbbs_dihedrals(self):
        """
        Calculates the sbbs dihedral angles.

        Returns:
            Numpy array of dihedral angles between the backbone atoms of
            the units in the chain.
        """

        # The chain should be a protein chain
        assert self.chain_type == "protein"
        sbbs_dihedrals = []
        for i in range(len(self.units) - 1):
            if (
                self.units[i].unit_type == "type_2"
                and self.units[i + 1].unit_type == "type_2"
            ):
                ca_1 = self.units[i].coms["ca_coords"]
                sc_com_1 = self.units[i].coms["sidechain_com"]
                ca_2 = self.units[i + 1].coms["ca_coords"]
                sc_com_2 = self.units[i + 1].coms["sidechain_com"]
                sbbs_dihedrals.append(dihedral(sc_com_1, ca_1, ca_2, sc_com_2))

        return np.array(sbbs_dihedrals)

    def get_sbbb_dihedrals(self):
        """
        Calculates the sbbb dihedral angles.

        Returns:
            dataframe of dihedral angles between the backbone atoms of
            the units in the chain.
        """

        # The chain should be a protein chain
        assert self.chain_type == "protein"
        sbbb_dihedrals = {
            "resname": [],
            "dihedral": [],
        }
        for i in range(len(self.units) - 2):
            if self.units[i].unit_type == "type_2":
                sc_com_1 = self.units[i].coms["sidechain_com"]
                ca_1 = self.units[i].coms["ca_coords"]
                ca_2 = self.units[i + 1].coms["ca_coords"]
                ca_3 = self.units[i + 2].coms["ca_coords"]
                sbbb_dihedrals["dihedral"].append(dihedral(sc_com_1, ca_1, ca_2, ca_3))
                sbbb_dihedrals["resname"].append(self.units[i].resname)

        return pd.DataFrame(sbbb_dihedrals)

    def get_bbbs_dihedrals(self):
        """
        Calculates the bbbs dihedral angles.

        Returns:
            dataframe of dihedral angles between the backbone atoms of
            the units in the chain.
        """

        # The chain should be a protein chain
        assert self.chain_type == "protein"
        bbbs_dihedrals = {
            "resname": [],
            "dihedral": [],
        }
        for i in range(len(self.units) - 2):
            if self.units[i + 2].unit_type == "type_2":
                ca_1 = self.units[i].coms["ca_coords"]
                ca_2 = self.units[i + 1].coms["ca_coords"]
                ca_3 = self.units[i + 2].coms["ca_coords"]
                sc_com_3 = self.units[i + 2].coms["sidechain_com"]
                bbbs_dihedrals["dihedral"].append(dihedral(ca_1, ca_2, ca_3, sc_com_3))
                bbbs_dihedrals["resname"].append(self.units[i + 1].resname)

        return pd.DataFrame(bbbs_dihedrals)
