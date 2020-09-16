"""Tests data processing functionality in src/aposteriori/create_frame_dataset.py"""
from pathlib import Path
import copy
import tempfile

from hypothesis import given, settings
from hypothesis.strategies import integers
import ampal
import ampal.geometry as g
import aposteriori.data_prep.create_frame_data_set as cfds
import h5py
import numpy as np
import numpy.testing as npt
import pytest

TEST_DATA_DIR = Path("tests/testing_files/pdb_files/")


@settings(deadline=1500)
@given(integers(min_value=0, max_value=214))
def test_create_residue_frame_cnocb_encoding(residue_number):
    assembly = ampal.load_pdb(str(TEST_DATA_DIR / "3qy1.pdb"))
    focus_residue = assembly[0][residue_number]

    # Make sure that residue correctly aligns peptide plane to XY
    cfds.align_to_residue_plane(focus_residue)
    cfds.encode_cb_to_ampal_residue(focus_residue)
    assert np.array_equal(
        focus_residue["CA"].array, (0, 0, 0,)
    ), "The CA atom should lie on the origin."
    assert np.isclose(focus_residue["N"].x, 0), "The nitrogen atom should lie on XY."
    assert np.isclose(focus_residue["N"].z, 0), "The nitrogen atom should lie on XY."
    assert np.isclose(focus_residue["C"].z, 0), "The carbon atom should lie on XY."
    assert np.isclose(
        focus_residue["CB"].x, -0.741287356,
    ), f"The Cb has not been encoded at position X = -0.741287356"
    assert np.isclose(
        focus_residue["CB"].y, -0.53937931,
    ), f"The Cb has not been encoded at position Y = -0.53937931"
    assert np.isclose(
        focus_residue["CB"].z, -1.224287356,
    ), f"The Cb has not been encoded at position Z = -1.224287356"
    # Make sure that all relevant atoms are pulled into the frame
    frame_edge_length = 12.0
    voxels_per_side = 21
    centre = voxels_per_side // 2
    max_dist = np.sqrt(((frame_edge_length / 2) ** 2) * 3)
    for atom in (
        a
        for a in assembly.get_atoms(ligands=False)
        if cfds.within_frame(frame_edge_length, a)
    ):
        assert g.distance(atom, (0, 0, 0)) <= max_dist, (
            "All atoms filtered by `within_frame` should be within "
            "`frame_edge_length/2` of the origin"
        )
    # Obtain atom encoder:
    codec = cfds.Codec.CNOCB()
    # Make sure that aligned residue sits on XY after it is discretized
    single_res_assembly = ampal.Assembly(
        molecules=ampal.Polypeptide(monomers=copy.deepcopy(focus_residue).backbone)
    )
    # Need to reassign the parent so that the residue is the only thing in the assembly
    single_res_assembly[0].parent = single_res_assembly
    single_res_assembly[0][0].parent = single_res_assembly[0]
    array = cfds.create_residue_frame(
        single_res_assembly[0][0], frame_edge_length, voxels_per_side, encode_cb=True, codec=codec)
    np.testing.assert_array_equal(array[centre, centre, centre], [True, False, False, False], err_msg="The central atom should be CA.")
    nonzero_indices = list(zip(*np.nonzero(array)))
    assert (
        len(nonzero_indices) == 5
    ), "There should be only 5 backbone atoms in this frame"
    nonzero_on_xy_indices = list(zip(*np.nonzero(array[:, :, centre])))
    assert (
        3 <= len(nonzero_on_xy_indices) <= 4
    ), "N, CA and C should lie on the xy plane."


@settings(deadline=1500)
@given(integers(min_value=0, max_value=214))
def test_create_residue_frame_backbone_only(residue_number):
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
    frame_edge_length = 12.0
    voxels_per_side = 21
    centre = voxels_per_side // 2
    max_dist = np.sqrt(((frame_edge_length / 2) ** 2) * 3)
    for atom in (
        a
        for a in assembly.get_atoms(ligands=False)
        if cfds.within_frame(frame_edge_length, a)
    ):
        assert g.distance(atom, (0, 0, 0)) <= max_dist, (
            "All atoms filtered by `within_frame` should be within "
            "`frame_edge_length/2` of the origin"
        )

    # Make sure that aligned residue sits on XY after it is discretized
    single_res_assembly = ampal.Assembly(
        molecules=ampal.Polypeptide(monomers=copy.deepcopy(focus_residue).backbone)
    )
    # Need to reassign the parent so that the residue is the only thing in the assembly
    single_res_assembly[0].parent = single_res_assembly
    single_res_assembly[0][0].parent = single_res_assembly[0]
    # Obtain atom encoder:
    codec = cfds.Codec.CNO()
    array = cfds.create_residue_frame(
        single_res_assembly[0][0], frame_edge_length, voxels_per_side,
        encode_cb=False, codec=codec
    )
    np.testing.assert_array_equal(array[centre, centre, centre], [True, False, False], err_msg="The central atom should be CA.")
    nonzero_indices = list(zip(*np.nonzero(array)))
    assert (
        len(nonzero_indices) == 4
    ), "There should be only 4 backbone atoms in this frame"
    nonzero_on_xy_indices = list(zip(*np.nonzero(array[:, :, centre])))
    assert (
        3 <= len(nonzero_on_xy_indices) <= 4
    ), "N, CA and C should lie on the xy plane."


@given(integers(min_value=1))
def test_even_voxels_per_side(voxels_per_side):
    frame_edge_length = 18.0
    if voxels_per_side % 2:
        voxels_per_side += 1
    # Obtain atom encoder:
    codec = cfds.Codec.CNO()
    with pytest.raises(AssertionError, match=r".*must be odd*"):
        output_file_path = cfds.make_frame_dataset(
            structure_files=["eep"],
            output_folder=".",
            name="test_dataset",
            frame_edge_length=18.0,
            voxels_per_side=voxels_per_side,
            require_confirmation=False,
            encode_cb=True,
            codec=codec
        )


def test_make_frame_dataset():
    """Tests the creation of a frame data set."""
    test_file = TEST_DATA_DIR / "1ubq.pdb"
    frame_edge_length = 18.0
    voxels_per_side = 31

    ampal_1ubq = ampal.load_pdb(str(test_file))
    for atom in ampal_1ubq.get_atoms():
        if not cfds.default_atom_filter(atom):
            del atom.parent.atoms[atom.res_label]
            del atom
    with tempfile.TemporaryDirectory() as tmpdir:
        # Obtain atom encoder:
        codec = cfds.Codec.CNO()
        output_file_path = cfds.make_frame_dataset(
            structure_files=[test_file],
            output_folder=tmpdir,
            name="test_dataset",
            frame_edge_length=frame_edge_length,
            voxels_per_side=voxels_per_side,
            verbosity=1,
            require_confirmation=False,
            codec=codec,
        )
        with h5py.File(output_file_path, "r") as dataset:
            for n in range(1, 77):
                # check that the frame for all the data frames match between the input
                # arrays and the ones that come out of the HDF5 data set
                residue_number = str(n)
                test_frame = cfds.create_residue_frame(
                    residue=ampal_1ubq["A"][residue_number],
                    frame_edge_length=frame_edge_length,
                    voxels_per_side=voxels_per_side,
                    encode_cb=False,
                    codec=codec,
                )
                hdf5_array = dataset["1ubq"]["A"][residue_number][()]
                npt.assert_array_equal(
                    hdf5_array,
                    test_frame,
                    err_msg=(
                        "The frame in the HDF5 data set should be the same as the "
                        "input frame."
                    ),
                )


@settings(deadline=700)
@given(integers(min_value=0, max_value=214))
def test_default_atom_filter(residue_number: int):
    assembly = ampal.load_pdb(str(TEST_DATA_DIR / "3qy1.pdb"))
    focus_residue = assembly[0][residue_number]
    backbone_atoms = ("N", "CA", "C", "O")

    for atom in focus_residue:
        filtered_atom = True if atom.res_label in backbone_atoms else False
        filtered_scenario = cfds.default_atom_filter(atom)
        assert filtered_atom == filtered_scenario, f"Expected {atom.res_label} to return {filtered_atom} after filter"


@settings(deadline=700)
@given(integers(min_value=0, max_value=214))
def test_cb_atom_filter(residue_number: int):
    assembly = ampal.load_pdb(str(TEST_DATA_DIR / "3qy1.pdb"))
    focus_residue = assembly[0][residue_number]
    backbone_atoms = ("N", "CA", "C", "O", "CB")

    for atom in focus_residue:
        filtered_atom = True if atom.res_label in backbone_atoms else False
        filtered_scenario = cfds.keep_sidechain_cb_atom_filter(atom)
        assert filtered_atom == filtered_scenario, f"Expected {atom.res_label} to return {filtered_atom} after filter"


def test_download_pdb_from_csv_file():
    csv_test_file = Path("tests/testing_files/csv_pdb_list/pdb_to_test.csv")
    test_file_paths = cfds.download_pdb_from_csv_file(csv_test_file, TEST_DATA_DIR)
    assert TEST_DATA_DIR / "1qys.pdb1" in test_file_paths, f"Expected to find {TEST_DATA_DIR / '1qys.pdb1'} as part of the generated paths."
    assert TEST_DATA_DIR / "1qys.pdb1" in test_file_paths, f"Expected to find {TEST_DATA_DIR / '3qy1A.pdb1'} as part of the generated paths."
    assert TEST_DATA_DIR / "6ct4.pdb1" in test_file_paths, f"Expected to find {TEST_DATA_DIR / '6ct4.pdb1'} as part of the generated paths."
    assert (TEST_DATA_DIR / "1qys.pdb1").exists(), f"Expected download of 1QYS to return PDB file"
    assert (TEST_DATA_DIR / "3qy1A.pdb1").exists(), f"Expected download of 1QYS to return PDB file"
    assert (TEST_DATA_DIR / "6ct4.pdb1").exists(), f"Expected download of 6CT4 to return PDB file"
