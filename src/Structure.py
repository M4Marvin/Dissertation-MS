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

    pdb_id: str
    structure: BioStructure
    protein_chains: List[Chain] = None
    ssdna_chains: List[Chain] = None
    dfs: List[pd.DataFrame] = None

    def __post_init__(self):
        """
        Initializes the chains and DataFrames attributes post object
        instantiation.
        """
        _chains = self._generate_chain_objects()
        self.protein_chains = [
            chain for chain in _chains if chain.chain_type == "protein"
        ]
        self.ssdna_chains = [chain for chain in _chains if chain.chain_type == "ssdna"]
        self.dfs = self._convert_chains_to_dataframes()

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
        return [chain.df for chain in self.chains]

    def save_as_markdown(
        self,
        out_path: str = "data/markdowns",
        show: bool = True,
    ) -> None:
        """
        Saves the structure as a markdown file.

        Args:
            out_path : str, optional
                Path to save the markdown file to
            show : bool, optional
                Whether to show the markdown file in the browser
        """

        out_file = f"{out_path}/{self.pdb_id}.md"

        with open(out_file, "w") as f:
            f.write(f"# {self.pdb_id}\n\n")
            f.write(f"## Number of chains: {len(self.chains)}\n\n")

            for i, chain in enumerate(self.chains):
                f.write(f"### Chain {i + 1}\n\n")
                f.write(chain.df.to_markdown())

        if show:
            import webbrowser

            webbrowser.open(out_file)

    def get_protein_distances(self):
        """
        Returns a DataFrame with the combined distances between the different
        chains in the structure. Two distances are calculated:
        1) Backbone to backbone
        2) Backbone to sidechain COM

        Returns:
            A tuple of a np array and a DataFrame with the distances between
            the different residues in the structure.
        """

        bb_distances = np.concatenate(
            [chain.get_protein_distances_bb() for chain in self.protein_chains],
            axis=1,
        )

        df_bs = pd.concat(
            [chain.get_protein_distances_bs() for chain in self.protein_cchains],
            axis=1,
        )

        return bb_distances, df_bs

    def get_protein_angles(self):
        """
        Returns a DataFrame with the combined angles between the different
        chains in the structure. 3 angles are calculated:
        1) BBB
        2) BBS
        3) SBB

        Returns:
            A tuple of one np.array and two DataFrames with the angles between
            the different chains in the structure.
        """

        bbb_arrays, dfs_bbs, dfs_sbb = [], [], []

        for chain in self.protein_chains:
            bbb, bbs, sbb = chain.get_protein_angles()
            bbb_array.append(bbb)
            df_bbs.append(bbs)
            df_sbb.append(sbb)

        bbb_array = np.concatenate(bbb_arrays, axis=1)
        dfs_bbs = pd.concat(df_bbs, axis=1)
        dfs_sbb = pd.concat(df_sbb, axis=1)

        return df_bbb, df_bbs, df_sbb

    def get_protein_dihedrals(self):
        """
        Returns a DataFrame with the combined dihedrals between the different
        chains in the structure. 4 dihedrals are calculated:
        1) BBBB
        2) SBBS
        3) SBBB
        4) BBBS

        Returns:
            A tuple of one np.array and four DataFrames with the dihedrals
            between the different chains in the structure.
        """

        df_bbbb, df_bbbs, df_bbsb, df_bsbb = [], [], [], []

        for chain in self.protein_chains:
            bbbb, bbbs, bbsb, bsbb = chain.get_protein_dihedrals()
            df_bbbb.append(bbbb)
            df_bbbs.append(bbbs)
            df_bbsb.append(bbsb)
            df_bsbb.append(bsbb)

        df_bbbb = pd.concat(df_bbbb, axis=1)
        df_bbbs = pd.concat(df_bbbs, axis=1)
        df_bbsb = pd.concat(df_bbsb, axis=1)
        df_bsbb = pd.concat(df_bsbb, axis=1)

        return df_bbbb, df_bbbs, df_bbsb, df_bsbb

    def get_ssdna_distances(self):
        """
        Returns a DataFrame with the combined distances between the different
        chains in the structure. Two distances are calculated:
        1) Backbone to backbone
        2) Backbone to sidechain COM

        Returns:
            A tuple of DataFrames with the distances between the different
            chains in the structure.
        """
        df_distances, df_bb_distances = pd.DataFrame(), pd.DataFrame()
        for chain in self.ssdna_chains:
            df_dist, df_bb_dist = chain.get_nu_distances()
            df_distances = pd.concat([df_distances, df_dist], axis=1)
            df_bb_distances = pd.concat([df_bb_distances, df_bb_dist], axis=1)

        return df_distances, df_bb_distances

    def get_ssdna_angles(self):
        """
        Returns a DataFrame with the combined angles between the different
        chains in the structure. 4 angles are calculated:
        1) PSB
        2) PSP
        3) BSP
        4) SPS

        Returns:
            A dataframe with the angles between the different chains in the
            structure.
        """
        df_angles = pd.concat(
            [chain.get_nu_angles() for chain in self.ssdna_chains],
            axis=1,
        )

        return df_angles

    def get_ssdna_dihedrals(self):
        """
        Returns a DataFrame with the combined dihedrals between the different
        chains in the structure. 2 dihedrals are calculated:
        1) PSPS
        2) SPSP

        Returns:
            A tuple of dataframes with the dihedrals between the different
            chains in the structure.
        """
        df_psps, df_spsp = [], []

        for chain in self.ssdna_chains:
            psps, spsp = chain.get_nu_dihedrals()
            df_psps.append(psps)
            df_spsp.append(spsp)

        df_psps = pd.concat(df_psps, axis=1)
        df_spsp = pd.concat(df_spsp, axis=1)

        return df_psps, df_spsp

    # A method that iterates over all chains and Stores the dataframes
    # and arrays as attributes of the class

    def get_all_data(self):
        """
        Iterates over all chains and stores the dataframes and arrays as
        attributes of the class.
        """

        self.df_bb, self.df_bs = self.get_protein_distances()
        self.df_bbb, self.df_bbs, self.df_sbb = self.get_protein_angles()
        (
            self.df_bbbb,
            self.df_bbbs,
            self.df_bbsb,
            self.df_bsbb,
        ) = self.get_protein_dihedrals()
        self.df_distances, self.df_bb_distances = self.get_ssdna_distances()
        self.df_angles = self.get_ssdna_angles()
        self.df_psps, self.df_spsp = self.get_ssdna_dihedrals()
