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
