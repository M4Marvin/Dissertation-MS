from src.ChainUnit import ChainUnit
from src.utils import ChainTypeHelper
from Bio.PDB.Chain import Chain as BioChain
from typing import List
import pandas as pd
from dataclasses import dataclass


@dataclass
class Chain:
    chain: BioChain
    chain_type: str = None
    units: List[ChainUnit] = None
    chain_df: pd.DataFrame = None

    def __post_init__(self):
        self.chain_type = ChainTypeHelper.get_chain_type(self.chain)
        self.units = self._generate_units()
        self.chain_df = self._create_dataframe()

    def _generate_units(self) -> List[ChainUnit]:
        return [ChainUnit(unit) for unit in self.chain.get_residues()]

    def __str__(self) -> str:
        out = (
            f"Chain type: "
            f"{self.chain_type}\n"
            f"Number of units: {len(self.units)}\n"
        )
        return out

    def _create_dataframe(self) -> pd.DataFrame:
        unit_dicts = [unit.to_dict() for unit in self.units if unit.fidelity]
        df = pd.DataFrame(unit_dicts)
        return df
