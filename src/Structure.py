from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd
from Bio.PDB.Chain import Chain as BioChain
from Bio.PDB.Structure import Structure as BioStructure
from Bio.PDB import PDBParser

from src.Chain import ChainGenerator, Chain
from src.utils import combine_dicts, print_key_shapes


@dataclass
class ChainHandler:
    structure: BioStructure

    def generate_chains(self) -> Tuple[List[Chain], List[Chain]]:
        """
        Generates Chain objects from the BioPython structure.
        Separates them into protein and ssdna chains.

        Returns:
            Tuple containing lists of protein and ssdna Chain objects.
        """
        chains = []
        for chain in self.structure.get_chains():
            try:
                chains.append(ChainGenerator.generate_chain(chain))
                # print(f"Chain {chain.id} generated successfully.")
            except Exception as e:
                print(e)
                print(f"Chain {chain.id} could not be generated. Skipping...")
                raise e
            # finally:
            #     print(f"Chain {chain.id} generated successfully.")

        protein_chains = [chain for chain in chains if chain.type == "protein"]
        ssdna_chains = [chain for chain in chains if chain.type == "ssdna"]

        return protein_chains, ssdna_chains

    def merge_calculations(self, chains: List[Chain]) -> Dict:
        """
        Merges the calculations of the given chains.

        Returns:
            A dictionary containing the merged calculations.
        """
        distances = [chain.distances for chain in chains]
        angles = [chain.angles for chain in chains]
        dihedrals = [chain.dihedrals for chain in chains]

        return {
            "distances": combine_dicts(distances),
            "angles": combine_dicts(angles),
            "dihedrals": combine_dicts(dihedrals),
        }

    def write_chains_to_file(self, file, chains: List[Chain], header: str) -> None:
        file.write(f"## {header}\n\n")
        for i, chain in enumerate(chains):
            file.write(f"\n### Chain {i + 1}\n\n")
            file.write(chain.df.to_markdown())


@dataclass
class Structure:
    """
    Class Structure to represent a biopolymer structure from a PDB file.

    Attributes:
        pdb_id : str
            PDB ID of the structure.
        structure : BioStructure
            A BioPython structure object.
        protein_chains : List[Chain], optional
            List of Chain objects representing the protein chains in the structure.
        ssdna_chains : List[Chain], optional
            List of Chain objects representing the ssDNA chains in the structure.
        protein_calculations : Dict, optional
            Dictionary containing the merged calculations for protein chains.
        ssdna_calculations : Dict, optional
            Dictionary containing the merged calculations for ssDNA chains.
    """

    pdb_id: str
    structure: BioStructure
    protein_chains: List[Chain] = field(init=False)
    ssdna_chains: List[Chain] = field(init=False)
    protein_calculations: Dict = field(init=False)
    ssdna_calculations: Dict = field(init=False)
    header: str = field(init=False)

    def __post_init__(self):
        """
        Generates Chain objects from the BioPython structure.
        Separates them into protein and ssdna chains.
        """
        chain_handler = ChainHandler(self.structure)
        self.protein_chains, self.ssdna_chains = chain_handler.generate_chains()
        self.protein_calculations = chain_handler.merge_calculations(
            self.protein_chains
        )
        self.ssdna_calculations = chain_handler.merge_calculations(self.ssdna_chains)

    def __str__(self) -> str:
        """
        Returns a string representation of the structure.

        Returns:
            A string representation of the structure.
        """

        number_of_chains = (
            f"Number of chains: {len(self.protein_chains) + len(self.ssdna_chains)}\n"
        )
        protein_chains = "\nProtein chains:\n\n" + "\n".join(
            [f"Chain {i + 1}: {chain}" for i, chain in enumerate(self.protein_chains)]
        )
        ssdna_chains = "\nssDNA chains:\n\n" + "\n".join(
            [f"Chain {i + 1}: {chain}" for i, chain in enumerate(self.ssdna_chains)]
        )

        return number_of_chains + protein_chains + ssdna_chains

    def get_stats(self):
        # Return a dictionary with the number of chains of each type and
        # the shape of the dataframes/npy arrays for each chain
        stats = {}

        protein_stats = {}
        for _dict in self.protein_calculations.values():
            for key, value in _dict.items():
                protein_stats[key] = value.shape
        stats["Protein chains"] = protein_stats

        ssdna_stats = {}
        for _dict in self.ssdna_calculations.values():
            for key, value in _dict.items():
                ssdna_stats[key] = value.shape
        stats["ssDNA chains"] = ssdna_stats

        return stats

    def save_as_markdown(
        self, out_path: str = "data/markdowns", show: bool = True
    ) -> None:
        """
        Saves the structure as a markdown file.

        Args:
            out_path : str, optional
                Path to save the markdown file to.
            show : bool, optional
                Whether to show the markdown file in the browser.
        """
        out_file = f"{out_path}/{self.pdb_id}.md"

        with open(out_file, "w") as f:
            f.write(f"# {self.pdb_id}\n\n")
            f.write(
                f"## Number of chains: {len(self.protein_chains) + len(self.ssdna_chains)}\n\n"
            )

            chain_handler = ChainHandler(self.structure)
            chain_handler.write_chains_to_file(f, self.protein_chains, "Protein Chains")
            chain_handler.write_chains_to_file(f, self.ssdna_chains, "ssDNA Chains")

        if show:
            import webbrowser

            webbrowser.open(out_file)


# A structure generator that generates a structure from a PDB file path.
# The structure is generated from the PDB file using BioPython.


class StructureGenerator:
    @staticmethod
    def generate_structure(pdb_file_path: str) -> Structure:
        """
        Generates a structure from a PDB file path.

        Args:
            pdb_file_path : str
                Path to the PDB file.

        Returns:
            A Structure object.
        """
        pdb_id = pdb_file_path.split("/")[-1].split(".")[0]
        structure = PDBParser().get_structure(pdb_id, pdb_file_path)
        return Structure(pdb_id, structure)
