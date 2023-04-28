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
