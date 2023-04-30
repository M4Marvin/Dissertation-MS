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
        assert self.chain_type == "protein"
        bbs_angles = {
            "resname": [],
            "angle": [],
        }

        for i in range(len(self.units) - 1):
            if not self.units[i].fidelity:
                continue

            if self.units[i + 1].unit_type == "type_2":
                ca_1 = self.units[i].coms["ca_coords"]
                sc_com = self.units[i + 1].coms["sidechain_com"]
                ca_2 = self.units[i + 1].coms["ca_coords"]
                bbs_angles["angle"].append(angle(ca_1, ca_2, sc_com))
                bbs_angles["resname"].append(self.units[i].resname)

        return pd.DataFrame(bbs_angles)

    def get_protein_angles(self):
        """
        Calculates the angles between the backbone atoms of the units in the
        chain.

        Returns:
            Numpy array of angles between the backbone atoms of the units in
            the chain.
        """
        # The chain should be a protein chain
        assert self.chain_type == "protein"
        return (self.get_bbb_angles(), self.get_sbb_angles(), self.get_bbs_angles())

    def get_bbbb_dihedrals(self):
        """
        Calculates the bbbb dihedral angles.

        Returns:
            Numpy array of dihedral angles between the backbone atoms of
            the units in the chain.
        """
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

    def get_protein_dihedrals(self):
        """
        Calculates the dihedral angles between the backbone atoms of the units
        in the chain.

        Returns:
            Numpy array of dihedral angles between the backbone atoms of
            the units in the chain.
        """
        # The chain should be a protein chain
        assert self.chain_type == "protein"
        return (
            self.get_bbbb_dihedrals(),
            self.get_sbbs_dihedrals(),
            self.get_sbbb_dihedrals(),
            self.get_bbbs_dihedrals(),
        )

    def get_ssdna_distances(self):
        """
        Calculates the following distances between the nucleotide beads:
        1. Distance between the phosphate bead and the sugar bead.
        2. Distance between the sugar bead and the base bead.
        3. Distance between the sugar bead and the phosphate bead of the
              next nucleotide.
        4. Distance between the base bead and the base bead of the next
                nucleotide.

        Returns:
            Dataframe of the distances between the nucleotide beads.
        """

        # The chain should be a ssdna chain
        assert self.chain_type == "ssdna"
        distances = {
            "nu_name": [],
            "ps_distance": [],
            "sb_distance": [],
            "sp_distance": [],
        }
        bb_distances = {
            "nu_1_name": [],
            "nu_2_name": [],
            "bb_distance": [],
        }

        for i in range(len(self.units)):
            distances["nu_name"].append(self.units[i].resname)
            phosphate_xyz = self.units[i].coms["phosphate_com"]
            sugar_xyz = self.units[i].coms["sugar_com"]
            base_xyz = self.units[i].coms["base_com"]

            distances["ps_distance"].append(distance(phosphate_xyz, sugar_xyz))
            distances["sb_distance"].append(distance(sugar_xyz, base_xyz))

        for i in range(len(self.units) - 1):
            bb_distances["nu_1_name"].append(self.units[i].resname)
            bb_distances["nu_2_name"].append(self.units[i + 1].resname)
            next_phosphate_xyz = self.units[i + 1].coms["phosphate_com"]
            next_base_xyz = self.units[i + 1].coms["base_com"]
            bb_distances["bb_distance"].append(
                distance(next_phosphate_xyz, next_base_xyz)
            )
            distances["sp_distance"].append(distance(sugar_xyz, next_phosphate_xyz))

        distances["sp_distance"].append(np.nan)

        # Print the list lengths for distances
        # print("distances: ", len(distances["nu_name"]))
        # print("ps_distance: ", len(distances["ps_distance"]))
        # print("sb_distance: ", len(distances["sb_distance"]))
        # print("sp_distance: ", len(distances["sp_distance"]))

        # Print the list lengths for bb_distances
        # print("bb_distances: ", len(bb_distances["nu_1_name"]))
        # print("bb_distances: ", len(bb_distances["nu_2_name"]))
        # print("bb_distances: ", len(bb_distances["bb_distance"]))

        return pd.DataFrame(distances), pd.DataFrame(bb_distances)

    def get_ssdna_angles(self):
        """
        Calculates the following angles between the nucleotide beads:
        1. Angle between the phosphate bead, sugar bead and base bead.
        2. Angle between the phosphate bead, sugar bead and phosphate bead
              of the next nucleotide.
        3. Angle between the base bead, sugar bead and phosphate bead of the
              next nucleotide.
        4. Angle between the sugar bead, phosphate bead of the next nucleotide
              and the sugar bead of the next nucleotide.

        Returns:
            Dataframe of the angles between the nucleotide beads.
        """

        # The chain should be a ssdna chain
        assert self.chain_type == "ssdna"
        angles = {
            "nu_name": [],
            "psb_angle": [],
            "psp_angle": [],
            "bsp_angle": [],
            "sps_angle": [],
        }
        for i in range(len(self.units)):
            angles["nu_name"].append(self.units[i].resname)
            phosphate_xyz = self.units[i].coms["phosphate_com"]
            sugar_xyz = self.units[i].coms["sugar_com"]
            base_xyz = self.units[i].coms["base_com"]

            angles["psb_angle"].append(angle(phosphate_xyz, sugar_xyz, base_xyz))

        for i in range(len(self.units) - 1):
            if self.units[i + 1].fidelity and self.units[i].fidelity:
                phosphate_xyz = self.units[i].coms["phosphate_com"]
                sugar_xyz = self.units[i].coms["sugar_com"]
                base_xyz = self.units[i].coms["base_com"]
                next_phosphate_xyz = self.units[i + 1].coms["phosphate_com"]
                next_sugar_xyz = self.units[i + 1].coms["sugar_com"]
                angles["psp_angle"].append(
                    angle(phosphate_xyz, sugar_xyz, next_phosphate_xyz)
                )
                angles["bsp_angle"].append(
                    angle(base_xyz, sugar_xyz, next_phosphate_xyz)
                )
                angles["sps_angle"].append(
                    angle(sugar_xyz, next_phosphate_xyz, next_sugar_xyz)
                )
            else:
                angles["psp_angle"].append(np.nan)
                angles["bsp_angle"].append(np.nan)
                angles["sps_angle"].append(np.nan)

        angles["sps_angle"].append(np.nan)
        angles["bsp_angle"].append(np.nan)
        angles["psp_angle"].append(np.nan)

        # Print the list lengths for angles
        # print("angles: ", len(angles["nu_name"]))
        # print("psb_angle: ", len(angles["psb_angle"]))
        # print("psp_angle: ", len(angles["psp_angle"]))
        # print("bsp_angle: ", len(angles["bsp_angle"]))
        # print("sps_angle: ", len(angles["sps_angle"]))

        return pd.DataFrame(angles)

    def get_ssdna_dihedrals(self):
        """
        Calculates the following dihedrals between the nucleotide beads:
        1. psps dihedral that requires 2 nucleotides.
        2. spsp dihedral that requires 3 nucleotides.

        Returns:
            Tuple of numpy arrays of the dihedrals between the nucleotide beads.
        """

        # The chain should be a ssdna chain
        assert self.chain_type == "ssdna"
        psps_dihedrals = []
        spsp_dihedrals = []
        for i in range(len(self.units) - 1):
            if self.units[i + 1].fidelity and self.units[i].fidelity:
                phosphate_xyz = self.units[i].coms["phosphate_com"]
                sugar_xyz = self.units[i].coms["sugar_com"]
                next_phosphate_xyz = self.units[i + 1].coms["phosphate_com"]
                next_sugar_xyz = self.units[i + 1].coms["sugar_com"]
                psps_dihedrals.append(
                    dihedral(
                        phosphate_xyz, sugar_xyz, next_phosphate_xyz, next_sugar_xyz
                    )
                )

        for i in range(len(self.units) - 2):
            if (
                self.units[i + 2].fidelity
                and self.units[i + 1].fidelity
                and self.units[i].fidelity
            ):
                phosphate_xyz = self.units[i].coms["phosphate_com"]
                sugar_xyz = self.units[i].coms["sugar_com"]
                next_phosphate_xyz = self.units[i + 1].coms["phosphate_com"]
                next_sugar_xyz = self.units[i + 1].coms["sugar_com"]
                next_next_phosphate_xyz = self.units[i + 2].coms["phosphate_com"]
                spsp_dihedrals.append(
                    dihedral(
                        sugar_xyz,
                        next_phosphate_xyz,
                        next_sugar_xyz,
                        next_next_phosphate_xyz,
                    )
                )

        return np.array(psps_dihedrals), np.array(spsp_dihedrals)
