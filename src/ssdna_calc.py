from src.utils import distance, angle, dihedral
import numpy as np


def get_ps_distnaces(df):
    ps_distances = []
    for i in range(df.shape[0]):
        dist = distance(df["phosphate_com"][i], df["sugar_com"][i])
        ps_distances.append(dist)
    return ps_distances


def get_sp_distances(df):
    sp_distances = []
    for i in range(df.shape[0] - 1):
        dist = distance(df["sugar_com"][i], df["phosphate_com"][i + 1])
        sp_distances.append(dist)

    return sp_distances


def get_sb_distances(df):
    sb_distances = []
    for i in range(df.shape[0]):
        dist = distance(df["sugar_com"][i], df["base_com"][i])
        sb_distances.append(dist)
    return sb_distances


def get_bb_distances(df):
    bb_distances = []
    for i in range(df.shape[0] - 1):
        dist = distance(df["base_com"][i], df["base_com"][i + 1])
        bb_distances.append(dist)
    return bb_distances


def ssdna_distances(df):
    ps_distances = get_ps_distnaces(df)
    sp_distances = get_sp_distances(df)
    sb_distances = get_sb_distances(df)
    bb_distances = get_bb_distances(df)
    return ps_distances, sp_distances, sb_distances, bb_distances


def get_psb_angles(df):
    psb_angles = []
    for i in range(df.shape[0]):
        _angle = angle(df["phosphate_com"][i], df["sugar_com"][i], df["base_com"][i])
        psb_angles.append(_angle)
    return psb_angles


def get_psp_angles(df):
    psp_angles = []
    for i in range(df.shape[0] - 1):
        _angle = angle(
            df["phosphate_com"][i], df["sugar_com"][i], df["phosphate_com"][i + 1]
        )
        psp_angles.append(_angle)
    return psp_angles


def get_bsp_angles(df):
    bsp_angles = []
    for i in range(df.shape[0] - 1):
        _angle = angle(
            df["base_com"][i], df["sugar_com"][i], df["phosphate_com"][i + 1]
        )
        bsp_angles.append(_angle)
    return bsp_angles


def get_sps_angles(df):
    sps_angles = []
    for i in range(df.shape[0] - 1):
        _angle = angle(
            df["sugar_com"][i], df["phosphate_com"][i + 1], df["sugar_com"][i + 1]
        )
        sps_angles.append(_angle)
    return sps_angles


def get_ssdna_angles(df):
    psb_angles = get_psb_angles(df)
    psp_angles = get_psp_angles(df)
    bsp_angles = get_bsp_angles(df)
    sps_angles = get_sps_angles(df)
    return psb_angles, psp_angles, bsp_angles, sps_angles


def get_psps_dihedrals(df):
    psps_dihedral = []
    for i in range(df.shape[0] - 1):
        _dihedral = dihedral(
            df["phosphate_com"][i],
            df["sugar_com"][i],
            df["phosphate_com"][i + 1],
            df["sugar_com"][i + 1],
        )
        psps_dihedral.append(_dihedral)
    return psps_dihedral


def get_spsp_dihedrals(df):
    spsp_dihedral = []
    for i in range(df.shape[0] - 2):
        _dihedral = dihedral(
            df["sugar_com"][i],
            df["phosphate_com"][i + 1],
            df["sugar_com"][i + 1],
            df["phosphate_com"][i + 2],
        )
        spsp_dihedral.append(_dihedral)
    return spsp_dihedral


def get_ssdna_dihedrals(df):
    psps_dihedral = get_psps_dihedrals(df)
    spsp_dihedral = get_spsp_dihedrals(df)
    return psps_dihedral, spsp_dihedral
