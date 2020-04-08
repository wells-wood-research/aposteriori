"""Tests data processing functionality in src/aposteriori/create_frame_data_set.py"""
from pathlib import Path
import copy

import aposteriori.data_prep.create_frame_data_set as cfds
import ampal
import ampal.geometry as g
from hypothesis import given, settings
from hypothesis.strategies import integers
import numpy as np
import pytest

TEST_DATA_DIR = Path("tests/testing_files/pdb_files/")


@settings(deadline=400)
@given(integers(min_value=0, max_value=214))
def test_create_residue_frame(residue_number):
    assembly = ampal.load_pdb(str(TEST_DATA_DIR / "3qy1.pdb"))
    focus_residue = assembly[0][residue_number]

    # Make sure that residue correctly aligns peptide plane to XY
    cfds.align_to_residue_plane(focus_residue)
    assert np.array_equal(
        focus_residue["CA"].array, (0, 0, 0,)
    ), "The CA atom should lie on the origin."
    assert np.isclose(focus_residue["N"].x, 0), "The nitrogen atom should lie on XY."
    assert np.isclose(focus_residue["N"].z, 0), "The nitrogen atom should lie on XY."
    assert np.isclose(focus_residue["C"].z, 0), "The carbon atom should lie on XY."

    # Make sure that all relevant atoms are pulled into the frame
    radius = 6.0
    voxels_per_side = 21
    centre = voxels_per_side // 2
    max_dist = np.sqrt((radius ** 2) * 3)
    for atom in (
        a for a in assembly.get_atoms(ligands=False) if cfds.within_frame(radius, a)
    ):
        assert (
            g.distance(atom, (0, 0, 0)) <= max_dist
        ), "All atoms filtered by `within_frame` should be within `radius` of the origin"

    # Make sure that aligned residue sits on XY after it is discretized
    single_res_assembly = ampal.Assembly(
        molecules=ampal.Polypeptide(monomers=copy.deepcopy(focus_residue).backbone)
    )
    # Need to reassign the parent so that the residue is the only thing in the assembly
    single_res_assembly[0].parent = single_res_assembly
    single_res_assembly[0][0].parent = single_res_assembly[0]
    label, array = cfds.create_residue_frame(
        single_res_assembly[0][0], radius, voxels_per_side
    )
    assert array[centre, centre, centre] == 6, "The central atom should be CA."
    nonzero_indices = list(zip(*np.nonzero(array)))
    assert (
        len(nonzero_indices) == 4
    ), "There should be only 4 backbone atoms in this frame"
    nonzero_on_xy_indices = list(zip(*np.nonzero(array[:, :, centre])))
    assert (
        3 <= len(nonzero_on_xy_indices) <= 4
    ), "N, CA and C should lie on the xy plane."
