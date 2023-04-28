from Bio.PDB.Chain import Chain as BioChain
from src.ChainUnit import ChainUnit
from src.utils import ChainTypeHelper
from typing import List
from dataclasses import dataclass
import pandas as pd


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
        Initializes the chain_type, units and df attributes post object instantiation.
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
            f"Chain type: {self.chain_type}\n" f"Number of units: {len(self.units)}\n"
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
