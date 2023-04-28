from src.utils import distance, angle, dihedral
import numpy as np


def get_bb_distances(df):
    bb_distances = []
    for i in range(df.shape[0] - 1):
        dist = distance(df["ca_coords"][i], df["ca_coords"][i + 1])
        bb_distances.append(dist)
    return bb_distances


def get_bs_distances(df):
    bs_distances = []
    for i in range(df.shape[0]):
        if df["sidechain_coms"][i] is None:
            bs_distances.append(np.nan)
            continue
        dist = distance(df["ca_coords"][i], df["sidechain_coms"][i])
        bs_distances.append(dist)
    return bs_distances


def get_bbb_angles(df):
    bbb_angles = []
    for i in range(df.shape[0] - 2):
        _angle = angle(
            df["ca_coords"][i], df["ca_coords"][i + 1], df["ca_coords"][i + 2]
        )
        bbb_angles.append(_angle)
    return bbb_angles


def get_bbs_angles(df):
    bbs_angles = []
    for i in range(df.shape[0] - 1):
        if df["sidechain_coms"][i] is None:
            bbs_angles.append(np.nan)
            continue
        _angle = angle(
            df["ca_coords"][i], df["ca_coords"][i + 1], df["sidechain_coms"][i]
        )
        bbs_angles.append(_angle)
    return bbs_angles


def get_sbb_angles(df):
    sbb_angles = []
    for i in range(df.shape[0] - 1):
        if df["sidechain_coms"][i] is None:
            sbb_angles.append(np.nan)
            continue
        _angle = angle(
            df["sidechain_coms"][i], df["ca_coords"][i], df["ca_coords"][i + 1]
        )
        sbb_angles.append(_angle)
    return sbb_angles


def get_bbbb_dihedrals(df):
    bbbb_dihedrals = []
    for i in range(df.shape[0] - 3):
        _dihedral = dihedral(
            df["ca_coords"][i],
            df["ca_coords"][i + 1],
            df["ca_coords"][i + 2],
            df["ca_coords"][i + 3],
        )
        bbbb_dihedrals.append(_dihedral)
    return bbbb_dihedrals


def get_sbbs_dihedrals(df):
    sbbs_dihedrals = []
    for i in range(df.shape[0] - 1):
        if df["sidechain_coms"][i] is None or df["sidechain_coms"][i + 1] is None:
            sbbs_dihedrals.append(np.nan)
            continue
        _dihedral = dihedral(
            df["sidechain_coms"][i],
            df["ca_coords"][i],
            df["ca_coords"][i + 1],
            df["sidechain_coms"][i + 1],
        )
        sbbs_dihedrals.append(_dihedral)
    return sbbs_dihedrals


def get_sbbb_dihedrals(df):
    sbbb_dihedrals = []
    for i in range(df.shape[0] - 2):
        if df["sidechain_coms"][i] is None:
            sbbb_dihedrals.append(np.nan)
            continue
        _dihedral = dihedral(
            df["sidechain_coms"][i],
            df["ca_coords"][i],
            df["ca_coords"][i + 1],
            df["ca_coords"][i + 2],
        )
        sbbb_dihedrals.append(_dihedral)
    return sbbb_dihedrals


def get_bbbs_dihedrals(df):
    bbbs_dihedrals = []
    for i in range(df.shape[0] - 2):
        if df["sidechain_coms"][i + 2] is None:
            bbbs_dihedrals.append(np.nan)
            continue
        _dihedral = dihedral(
            df["ca_coords"][i],
            df["ca_coords"][i + 1],
            df["ca_coords"][i + 2],
            df["sidechain_coms"][i + 2],
        )
        bbbs_dihedrals.append(_dihedral)
    return bbbs_dihedrals


def get_protein_distances(df):
    bb_distances = get_bb_distances(df)
    bs_distances = get_bs_distances(df)

    return bb_distances, bs_distances


def get_protein_angles(df):
    bbb_angles = get_bbb_angles(df)
    bbs_angles = get_bbs_angles(df)
    sbb_angles = get_sbb_angles(df)

    return bbb_angles, bbs_angles, sbb_angles


def get_protein_dihedrals(df):
    bbbb_dihedrals = get_bbbb_dihedrals(df)
    sbbs_dihedrals = get_sbbs_dihedrals(df)
    sbbb_dihedrals = get_sbbb_dihedrals(df)
    bbbs_dihedrals = get_bbbs_dihedrals(df)

    return bbbb_dihedrals, sbbs_dihedrals, sbbb_dihedrals, bbbs_dihedrals


def get_protein_features(df):
    bb_distances, bs_distances = get_protein_distances(df)
    bbb_angles, bbs_angles, sbb_angles = get_protein_angles(df)
    (
        bbbb_dihedrals,
        sbbs_dihedrals,
        sbbb_dihedrals,
        bbbs_dihedrals,
    ) = get_protein_dihedrals(df)

    return (
        bb_distances,
        bs_distances,
        bbb_angles,
        bbs_angles,
        sbb_angles,
        bbbb_dihedrals,
        sbbs_dihedrals,
        sbbb_dihedrals,
        bbbs_dihedrals,
    )


if __name__ == "__main__":
    dfs, chain_types = pd.read_csv("data/1a0j.csv")
    (
        bb_distances,
        bs_distances,
        bbb_angles,
        bbs_angles,
        sbb_angles,
        bbbb_dihedrals,
        sbbs_dihedrals,
        sbbb_dihedrals,
        bbbs_dihedrals,
    ) = get_protein_features(dfs[0])
    print(bb_distances)
    print(bs_distances)
    print(bbb_angles)
    print(bbs_angles)
    print(sbb_angles)
    print(bbbb_dihedrals)
    print(sbbs_dihedrals)
    print(sbbb_dihedrals)
    print(bbbs_dihedrals)
