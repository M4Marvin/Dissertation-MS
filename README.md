# Dissertation - Marvin V. Prakash

This the code i wrote for my MS Dissertation, It involves loading and processing of Protein-SSDna complexes, then analysing the various quantities in an coarse-grain manner.

## Features

The process is comprised of multiple parts, which are:

- Preprocessing the pdb files by removing hydrogen molecules and HetAtoms, also broken chain atoms.
- Parsing of PDB files
- Identifying different unit types within a chain (protein, ssDNA, type_1, type_2)
- Calculating the center of mass for different unit types
- Validation of units based on the presence of necessary atoms
- Generation of Pandas DataFrames containing processed PDB data

---

## Installation

1. Clone the repository

   ```bash
   git clone https://github.com/M4Marvin/Dissertation-MS.git
   ```

2. Navigate to the project directory

   ```bash
   cd Dissertation-MS/
   ```

3. Create a new conda environment from the environment.yml file

   ```bash
   conda env create -f environment.yml
   ```

4. Activate the enviornment

   ```bash
    conda activate bio
   ```

## Usage

For usage instruction take a look at the notebooks I provided.
