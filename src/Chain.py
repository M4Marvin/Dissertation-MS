from typing import List

import numpy as np
import pandas as pd
from Bio.PDB.Chain import Chain as BioChain

from src.ChainUnit import ChainUnit, ChainUnitGenerator
from src.Point import angle, dihedral, distance
from src.utils import ChainTypeHelper

# Constants
TYPE_1 = "type_1"
TYPE_2 = "type_2"
CA_COORDS = "ca_coords"
SIDCHAIN_COM = "sidechain_com"


class Chain:
    """
    Class Chain to represent a chain in a biopolymer structure.

    Attributes:
        units : List[ChainUnit]
            List of ChainUnit objects representing the residues in the chain
    """

    def __init__(
        self, bio_chain: BioChain, chain_type: str, perform_calculations: bool = True
    ):
        self.chain = bio_chain
        self.type = chain_type
        self.units = self._generate_chain_units()

        # Trim the chain to remove none units from the ends
        self.units = remove_none_from_ends(self.units)

        self.distances = {}
        self.angles = {}
        self.dihedrals = {}

        if perform_calculations:
            self.calculate_distances()
            self.calculate_angles()
            self.calculate_dihedrals()

    def __str__(self) -> str:
        chain_info = f"Type: " f"{self.type}\n" f"Number of units: {len(self.units)}\n"
        # chain_info += f"Sequence: {self.chain.get_sequence()}\n"
        return chain_info

    def __repr__(self) -> str:
        return self.__str__()

    def trim_chain(self):
        """
        Trims the chain to remove none units from the ends.
        """
        for i, unit in enumerate(self.units):
            if unit is not None:
                break

        for j, unit in enumerate(self.units[::-1]):
            if unit is not None:
                break

        self.units = self.units[i : len(self.units) - j]

    def _generate_chain_units(self) -> List[ChainUnit]:
        """
        Generates chain units.

        Returns:
            List of ChainUnit objects.
        """
        gen = ChainUnitGenerator()
        return [
            gen.generate(unit) for unit in self.chain.get_residues() if unit is not None
        ]

    @property
    def df(self) -> pd.DataFrame:
        """
        Generates a dataframe from the units.

        Returns:
            A dataframe containing the units.
        """
        unit_dicts = [unit.to_dict() for unit in self.units if unit.fidelity]
        return pd.DataFrame(unit_dicts)

    def calculate_distances(self):
        raise NotImplementedError("Subclasses must implement this method")

    def calculate_angles(self):
        raise NotImplementedError("Subclasses must implement this method")

    def calculate_dihedrals(self):
        raise NotImplementedError("Subclasses must implement this method")


class ProteinChain(Chain):
    def __init__(self, chain: BioChain, perform_calculations: bool = True):
        super().__init__(chain, "protein", perform_calculations)

    def get_bb_distances(self) -> np.ndarray:
        """
        Calculates and stores the backbone distances.

        Returns:
            A numpy array containing the backbone distances.
        """
        self.distances["bb"] = np.array(
            [self._calculate_distance(i, i + 1) for i in range(len(self.units) - 1)]
        )

    def _calculate_distance(self, index1: int, index2: int) -> float:
        """
        Helper function to calculate distance.

        Args:
            index1: Index of the first unit.
            index2: Index of the second unit.

        Returns:
            The distance between the two units.
        """
        # ca_1 = self.units[index1].coms[CA_COORDS]
        # ca_2 = self.units[index2].coms[CA_COORDS]
        unit_1 = self.units[index1]
        unit_2 = self.units[index2]
        if unit_1 is None or unit_2 is None:
            return np.nan
        ca_1 = unit_1.coms[CA_COORDS]
        ca_2 = unit_2.coms[CA_COORDS]
        return distance(ca_1, ca_2)

    def get_bs_distances(self):
        """
        Calculates and stores the backbone-sidechain distances.
        """
        bs_data = {
            "resname": [],
            "distance": [],
        }
        for unit in self.units:
            if unit is not None:
                if unit.fidelity and unit.unit_type == TYPE_2:
                    ca = unit.coms[CA_COORDS]
                    sc_com = unit.coms[SIDCHAIN_COM]
                    bs_data["distance"].append(distance(ca, sc_com))
                    bs_data["resname"].append(unit.resname)

        self.distances["bs"] = pd.DataFrame(bs_data)

    def calculate_distances(self):
        """
        Calculates and stores the distances.
        """
        self.get_bb_distances()
        self.get_bs_distances()

    def get_bbb_angles(self):
        """
        Calculates and stores the backbone-backbone-backbone angles.
        """
        bbb_angles = []
        for i in range(len(self.units) - 2):
            ca_1 = self.units[i].coms["ca_coords"]
            ca_2 = self.units[i + 1].coms["ca_coords"]
            ca_3 = self.units[i + 2].coms["ca_coords"]
            bbb_angles.append(angle(ca_1, ca_2, ca_3))

        self.angles["bbb"] = np.array(bbb_angles)

    def get_sbb_angles(self):
        """
        Calculates and stores the sidechain-backbone-backbone angles.
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

        self.angles["sbb"] = pd.DataFrame(sbb_angles)

    def get_bbs_angles(self):
        """
        Calculates and stores the backbone-backbone-sidechain angles.
        """

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

        self.angles["bbs"] = pd.DataFrame(bbs_angles)

    def calculate_angles(self):
        """
        Calculates and stores the angles.
        """
        self.get_bbb_angles()
        self.get_sbb_angles()
        self.get_bbs_angles()

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

        self.dihedrals["bbbb"] = np.array(bbbb_dihedrals)

    def get_sbbs_dihedrals(self):
        """
        Calculates the sbbs dihedral angles.
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

        self.dihedrals["sbbs"] = np.array(sbbs_dihedrals)

    def get_sbbb_dihedrals(self):
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

        self.dihedrals["sbbb"] = pd.DataFrame(sbbb_dihedrals)

    def get_bbbs_dihedrals(self):
        """
        Calculates the bbbs dihedral angles.
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

        self.dihedrals["bbbs"] = pd.DataFrame(bbbs_dihedrals)

    def calculate_dihedrals(self):
        self.get_bbbb_dihedrals()
        self.get_sbbs_dihedrals()
        self.get_sbbb_dihedrals()
        self.get_bbbs_dihedrals()


class SSDNAChain(Chain):
    def __init__(self, chain, perform_calculations=False):
        super().__init__(chain, "ssdna", perform_calculations)

    def calculate_distances(self):
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
            sugar_xyz = self.units[i].coms["sugar_com"]

            bb_distances["bb_distance"].append(
                distance(next_phosphate_xyz, next_base_xyz)
            )
            distances["sp_distance"].append(distance(sugar_xyz, next_phosphate_xyz))

        distances["sp_distance"].append(np.nan)

        self.distances["ps_sb_sp"] = pd.DataFrame(distances)
        self.distances["bb"] = pd.DataFrame(bb_distances)

    def calculate_angles(self):
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

        self.angles["psb_psp_bsp_sps"] = pd.DataFrame(angles)

    def calculate_dihedrals(self):
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

        self.dihedrals["psps"] = pd.DataFrame(psps_dihedrals)
        self.dihedrals["spsp"] = pd.DataFrame(spsp_dihedrals)


class ChainGenerator:
    """
    Class that is used to generate a ssdna or protein chain from a given BioChain object.
    """

    @staticmethod
    def generate_chain(biochain: BioChain, perform_calculations: bool = True):
        """
        # Determine the type of chain and generate it.
        :param biochain: BioChain object
        :param perform_calculations: bool
        :return: None
        """

        chain_type = ChainTypeHelper.get_chain_type(biochain)

        # Generate the chain
        if chain_type == "ssdna":
            return SSDNAChain(biochain, perform_calculations)
        elif chain_type == "protein":
            return ProteinChain(biochain, perform_calculations)
        else:
            print(f"Unknown chain type: {chain_type}")


def remove_none_from_ends(lst):
    while lst and lst[-1] is None:
        lst.pop()
    while lst and lst[0] is None:
        lst.pop(0)
    return lst
