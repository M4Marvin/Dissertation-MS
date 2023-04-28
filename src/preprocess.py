import mdtraj as md
import os

from src.utils import get_chain_type


def remove_neg_resSeq(traj: md.Trajectory) -> md.Trajectory:
    """
    Removes atoms with negative resSeq values from a mdtraj.Trajectory object.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        The input trajectory.

    Returns
    -------
    mdtraj.Trajectory
        The output trajectory containing only atoms with positive resSeq values.
    """
    # Select all atoms with positive resSeq values
    positive_resSeq_atoms = traj.topology.select("resSeq < 9000")

    # Create a new trajectory containing only the atoms with positive resSeq values
    traj_no_neg_resSeq = traj.atom_slice(positive_resSeq_atoms)

    return traj_no_neg_resSeq


def remove_water(traj: md.Trajectory) -> md.Trajectory:
    """
    Removes water molecules from a mdtraj.Trajectory object.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        The input trajectory.

    Returns
    -------
    mdtraj.Trajectory
        The output trajectory containing only non-water atoms.
    """
    # Select all non-water atoms
    non_water_atoms = traj.topology.select("not water")

    # Create a new trajectory containing only the non-water atoms
    traj_no_water = traj.atom_slice(non_water_atoms)

    return traj_no_water


def remove_hydrogen(traj: md.Trajectory) -> md.Trajectory:
    """
    Removes hydrogen atoms from a mdtraj.Trajectory object.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        The input trajectory.

    Returns
    -------
    mdtraj.Trajectory
        The output trajectory containing only non-hydrogen atoms.
    """
    # Select all non-hydrogen atoms
    non_hydrogen_atoms = traj.topology.select("not element H")

    # Create a new trajectory containing only the non-hydrogen atoms
    traj_no_hydrogen = traj.atom_slice(non_hydrogen_atoms)

    return traj_no_hydrogen


def remove_extra_chains(traj):
    """
    Removes chains that are not protein or ssDNA from a mdtraj.Trajectory object.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        The input trajectory.

    Returns
    -------
    mdtraj.Trajectory
        The output trajectory containing only protein and ssDNA chains.
    """
    # Create a list of chains that are not protein or ssDNA
    chains_to_remove = []
    for chain in traj.topology.chains:
        if get_chain_type(chain) != "protein" and get_chain_type(chain) != "ssDNA":
            chains_to_remove.append(chain)

    # Remove the chains that are not protein or ssDNA
    traj_no_extra_chains = remove_chains(traj, chains_to_remove)

    return traj_no_extra_chains


def remove_chains(traj, chains_to_remove):
    """
    Removes chains from a mdtraj.Trajectory object.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        The input trajectory.

    chains_to_remove : list[mdtraj.Chain]
        The list of chains to remove.

    Returns
    -------
    mdtraj.Trajectory
        The output trajectory containing only the chains that were not removed.
    """

    atom_indices_not_to_remove = []
    for chain in traj.topology.chains:
        if chain not in chains_to_remove:
            for atom in chain.atoms:
                atom_indices_not_to_remove.append(atom.index)

    traj_no_chains = traj.atom_slice(atom_indices_not_to_remove)

    return traj_no_chains


def prepare_traj(
    traj: md.Trajectory,
    _remove_extra_chains=True,
    _remove_water=True,
    _remove_hydrogen=True,
    _remove_negative_indices=True,
):
    """
    Prepares a mdtraj.Trajectory object for analysis.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        The input trajectory.
    _remove_extra_chains : bool
        Whether to remove chains that are not protein or ssDNA.
    _remove_water : bool
        Whether to remove water molecules.
    _remove_hydrogen : bool
        Whether to remove hydrogen atoms.

    Returns
    -------
    mdtraj.Trajectory
        The output trajectory.

    """
    if _remove_extra_chains:
        traj = remove_extra_chains(traj)

    if _remove_water:
        traj = remove_water(traj)

    if _remove_hydrogen:
        traj = remove_hydrogen(traj)

    if _remove_negative_indices:
        traj = remove_neg_resSeq(traj)

    return traj


def clean_pdb(
    pdb_path: str,
    output_path: str,
    _remove_extra_chains=True,
    _remove_water=True,
    _remove_hydrogen=True,
    _remove_negative_indices=True,
) -> None:
    """
    Cleans a pdb file and saves the cleaned version.

    Parameters
    ----------
    pdb_path : str
        The path to the pdb file to clean.
    output_path : str
        The path to save the cleaned pdb file.
    _remove_extra_chains : bool
        Whether to remove chains that are not protein or ssDNA.
    _remove_water : bool
        Whether to remove water molecules.
    _remove_hydrogen : bool
        Whether to remove hydrogen atoms.
    _remove_negative_indices : bool
        Whether to remove atoms with negative resSeq values.
    """
    # Open the pdb file
    traj = md.load_pdb(pdb_path)

    # Prepare the trajectory
    traj = prepare_traj(
        traj,
        _remove_extra_chains=_remove_extra_chains,
        _remove_water=_remove_water,
        _remove_hydrogen=_remove_hydrogen,
        _remove_negative_indices=_remove_negative_indices,
    )

    # Save the trajectory
    traj.save(output_path)


def main():
    data_folder = "./data/ssDNA_binding_proteins_complex"
    file_name = "8DFA.pdb"
    file_path = os.path.join(data_folder, file_name)

    # Load the trajectory
    traj = md.load_pdb(file_path)

    # Prepare the trajectory
    traj = prepare_traj(traj)

    # Save the trajectory
    traj.save("./data/8DFA_clean.pdb")


if __name__ == "__main__":
    main()
