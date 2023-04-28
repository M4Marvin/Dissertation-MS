from Bio.PDB.Chain import Chain as BioChain
from Bio.PDB.Structure import Structure as BioStructure
from dataclasses import dataclass
from typing import List

import pandas as pd

from src.Chain import Chain


@dataclass
class Structure:
    """
    Class Structure to represent a biopolymer structure from a PDB file.

    Attributes:
        structure : BioStructure
            A BioPython structure object
        chains : List[Chain], optional
            List of Chain objects representing the chains in the structure
        dfs : List[pd.DataFrame], optional
            List of DataFrame objects representing the chains in the structure
    """

    structure: BioStructure
    chains: List[Chain] = None
    dfs: List[pd.DataFrame] = None

    def __post_init__(self):
        """
        Initializes the chains and DataFrames attributes post object
        instantiation.
        """
        self.chains = self._generate_chain_objects()
        self.dfs, self.chain_types = self._convert_chains_to_dataframes()

    def _generate_chain_objects(self) -> List[BioChain]:
        """
        Generates Chain objects from the BioPython structure.

        Returns:
            List of Chain objects representing each chain in the structure.
        """
        return [Chain(chain) for chain in self.structure.get_chains()]

    def __str__(self) -> str:
        """
        Returns a string representation of the Structure object.

        Returns:
            String representation of the Structure object.
        """
        number_of_chains = f"Number of chains: {len(self.chains)}\n"
        chains_data = " ".join(
            [f"{i:02d}. " + str(chain) for i, chain in enumerate(self.chains)]
        )

        return number_of_chains + chains_data

    def _convert_chains_to_dataframes(self) -> List[pd.DataFrame]:
        """
        Converts each Chain object in the structure to a pandas DataFrame.

        Returns:
            List of DataFrame objects representing each chain in the structure.
        """
        return [chain.to_dataframe() for chain in self.chains]
