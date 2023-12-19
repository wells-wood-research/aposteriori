"""Tests data processing functionality in src/aposteriori/create_frame_dataset.py"""
import copy
import tempfile
from pathlib import Path

import ampal
import ampal.geometry as g
import aposteriori.data_prep.create_frame_data_set as cfds
from aposteriori.data_prep.create_frame_data_set import default_atom_filter
import h5py
import numpy as np
import numpy.testing as npt
import pytest
from ampal.amino_acids import residue_charge, polarity_Zimmerman, standard_amino_acids
from hypothesis import given, settings
from hypothesis.strategies import integers

import aposteriori.data_prep.create_frame_data_set as cfds

TEST_DATA_DIR = Path("tests/testing_files/pdb_files/")


def test_cb_position():
    assembly = ampal.load_pdb(str(TEST_DATA_DIR / "3qy1.pdb"))
    frame_edge_length = 12.0
    voxels_per_side = 21
    codec = cfds.Codec.CNOCB()
    cfds.voxelise_assembly(
        assembly,
        name="3qy1",
        atom_filter_fn=default_atom_filter,
        frame_edge_length=frame_edge_length,
        voxels_per_side=voxels_per_side,
        encode_cb=True,
        codec=codec,
        tag_rotamers=False,
        chain_dict={},
        voxels_as_gaussian=False,
        verbosity=1,
        chain_filter_list=["A", "B"],
    )

    for chain in assembly:
        for residue in chain:
            if not isinstance(residue, ampal.Residue):
                continue
            cfds.align_to_residue_plane(residue)
            assert np.isclose(
                residue["CB"].x,
                (residue["CA"].x - 0.741287356),
            ), f"The Cb has not been encoded at position X = -0.741287356"
            assert np.isclose(
                residue["CB"].y,
                (residue["CA"].y - 0.53937931),
            ), f"The Cb has not been encoded at position Y = -0.53937931"
            assert np.isclose(
                residue["CB"].z,
                (residue["CA"].z - 1.224287356),
            ), f"The Cb has not been encoded at position Z = -1.224287356"


@settings(deadline=1500)
@given(integers(min_value=0, max_value=214))
def test_create_residue_frame_cnocb_encoding(residue_number):
    assert (TEST_DATA_DIR / "3qy1.pdb").exists(), "File does not exist"
    assembly = ampal.load_pdb(str(TEST_DATA_DIR / "3qy1.pdb"))
    focus_residue = assembly[0][residue_number]

    # Make sure that residue correctly aligns peptide plane to XY
    cfds.encode_cb_prevox(focus_residue)
    assert np.array_equal(
        focus_residue["CA"].array,
        (
            0,
            0,
            0,
        ),
    ), "The CA atom should lie on the origin."
    assert np.isclose(focus_residue["N"].x, 0), "The nitrogen atom should lie on XY."
    assert np.isclose(focus_residue["N"].z, 0), "The nitrogen atom should lie on XY."
    assert np.isclose(focus_residue["C"].z, 0), "The carbon atom should lie on XY."
    assert np.isclose(
        focus_residue["CB"].x,
        -0.741287356,
    ), f"The Cb has not been encoded at position X = -0.741287356"
    assert np.isclose(
        focus_residue["CB"].y,
        -0.53937931,
    ), f"The Cb has not been encoded at position Y = -0.53937931"
    assert np.isclose(
        focus_residue["CB"].z,
        -1.224287356,
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
        molecules=ampal.Polypeptide(
            monomers=copy.deepcopy(focus_residue).backbone, polymer_id="A"
        )
    )
    # Need to reassign the parent so that the residue is the only thing in the assembly
    single_res_assembly[0].parent = single_res_assembly
    single_res_assembly[0][0].parent = single_res_assembly[0]
    chaindict = cfds.voxelise_assembly(
        single_res_assembly[0][0].parent.parent,
        name="3qy1",
        atom_filter_fn=default_atom_filter,
        frame_edge_length=frame_edge_length,
        voxels_per_side=voxels_per_side,
        encode_cb=True,
        codec=codec,
        tag_rotamers=False,
        chain_dict={},
        voxels_as_gaussian=False,
        verbosity=1,
        chain_filter_list=["A"],
    )[1]
    array_test = chaindict["A"][0].data
    np.testing.assert_array_equal(
        array_test[centre, centre, centre],
        [True, False, False, False],
        err_msg="The central atom should be CA.",
    )
    nonzero_indices = list(zip(*np.nonzero(array_test)))
    assert (
        len(nonzero_indices) == 5
    ), "There should be only 5 backbone atoms in this frame"
    nonzero_on_xy_indices = list(zip(*np.nonzero(array_test[:, :, centre])))
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
        focus_residue["CA"].array,
        (
            0,
            0,
            0,
        ),
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
        single_res_assembly[0][0],
        frame_edge_length,
        voxels_per_side,
        encode_cb=False,
        codec=codec,
    )
    np.testing.assert_array_equal(
        array[centre, centre, centre],
        [True, False, False],
        err_msg="The central atom should be CA.",
    )
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
            frame_edge_length=frame_edge_length,
            voxels_per_side=voxels_per_side,
            require_confirmation=False,
            encode_cb=True,
            codec=codec,
        )


def test_make_frame_dataset():
    """Tests the creation of a frame data set."""
    test_file = TEST_DATA_DIR / "1ubq.pdb"
    frame_edge_length = 18.0
    voxels_per_side = 31

    ampal_1ubq = ampal.load_pdb(str(test_file))
    if isinstance(ampal_1ubq, ampal.AmpalContainer):
        for assembly in ampal_1ubq:
            for monomer in assembly:
                if isinstance(monomer, ampal.Polypeptide):
                    monomer.tag_sidechain_dihedrals()
    elif isinstance(ampal_1ubq, ampal.Polypeptide):
        ampal_1ubq.tag_sidechain_dihedrals()
    elif isinstance(ampal_1ubq, ampal.Assembly):
        # For each monomer in the assembly:
        for monomer in ampal_1ubq:
            if isinstance(monomer, ampal.Polypeptide):
                monomer.tag_sidechain_dihedrals()
    else:
        raise ValueError

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
            tag_rotamers=True,
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
                rota = ""
                for r in ampal_1ubq[0][n - 1].tags["rotamers"]:
                    rota += str(r)
                assert (
                    rota == dataset["1ubq"]["A"][residue_number].attrs["rotamers"]
                ), f'Tags Rotamer mismatch found at position {n}: {dataset["1ubq"]["A"][residue_number].attrs["rotamers"]} but expected {rota}'
                hdf5_array = dataset["1ubq"]["A"][residue_number][()]
                npt.assert_array_equal(
                    hdf5_array,
                    test_frame,
                    err_msg=(
                        "The frame in the HDF5 data set should be the same as the "
                        "input frame."
                    ),
                )


def test_convert_atom_to_gaussian_density():
    # No modifiers:
    opt_frame = cfds.convert_atom_to_gaussian_density((0, 0, 0), 0.6, optimized=True)
    non_opt_frame = cfds.convert_atom_to_gaussian_density(
        (0, 0, 0), 0.6, optimized=False
    )
    np.testing.assert_array_almost_equal(opt_frame, non_opt_frame, decimal=2)
    np.testing.assert_almost_equal(np.sum(non_opt_frame), np.sum(opt_frame))
    # With modifiers:
    opt_frame = cfds.convert_atom_to_gaussian_density((0.5, 0, 0), 0.6, optimized=True)
    non_opt_frame = cfds.convert_atom_to_gaussian_density(
        (0.5, 0, 0), 0.6, optimized=False
    )
    np.testing.assert_array_almost_equal(opt_frame, non_opt_frame, decimal=2)


def test_make_frame_dataset_as_gaussian():
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
            voxels_as_gaussian=True,
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
                    voxels_as_gaussian=True,
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


def test_make_frame_dataset_as_gaussian_cnocacbq():
    """Tests the creation of a frame data set."""
    test_file = TEST_DATA_DIR / "1ubq.pdb"
    frame_edge_length = 18.0
    voxels_per_side = 31
    codec = cfds.Codec.CNOCACBQ()
    ampal_1ubq = ampal.load_pdb(str(test_file))
    ampal_1ubq2 = ampal.load_pdb(str(test_file))

    test_frame = cfds.voxelise_assembly(
        ampal_1ubq2,
        name="1ubq",
        atom_filter_fn=default_atom_filter,
        frame_edge_length=frame_edge_length,
        voxels_per_side=voxels_per_side,
        encode_cb=True,
        codec=codec,
        tag_rotamers=False,
        chain_dict={},
        voxels_as_gaussian=True,
        verbosity=1,
        chain_filter_list=["A"],
    )[1]

    array_test = []
    for k in range(76):
        array_test.append(test_frame["A"][k].data)

    for atom in ampal_1ubq.get_atoms():
        if not cfds.default_atom_filter(atom):
            del atom.parent.atoms[atom.res_label]
            del atom
    with tempfile.TemporaryDirectory() as tmpdir:
        # Obtain atom encoder:
        output_file_path = cfds.make_frame_dataset(
            structure_files=[test_file],
            output_folder=tmpdir,
            name="test_dataset",
            frame_edge_length=frame_edge_length,
            voxels_per_side=voxels_per_side,
            verbosity=1,
            require_confirmation=False,
            codec=codec,
            voxels_as_gaussian=True,
        )
        with h5py.File(output_file_path, "r") as dataset:
            for n in range(1, 77):
                # check that the frame for all the data frames match between the input
                # arrays and the ones that come out of the HDF5 data set
                residue_number = str(n)
                test_residue = array_test[n - 1]
                hdf5_array = dataset["1ubq"]["A"][residue_number][()]
                npt.assert_array_equal(
                    hdf5_array,
                    test_residue,
                    err_msg=(
                        "The frame in the HDF5 data set should be the same as the "
                        "input frame."
                    ),
                )
                charge = residue_charge[ampal_1ubq["A"][residue_number].mol_letter]
                if charge > 0:
                    assert np.max(test_residue[:, :, :, 5]) > 0
                if charge < 0:
                    assert np.min(test_residue[:, :, :, 5]) < 0


def test_make_frame_dataset_as_gaussian_cnocacbp():
    """Tests the creation of a frame data set."""
    test_file = TEST_DATA_DIR / "1ubq.pdb"
    frame_edge_length = 18.0
    voxels_per_side = 31
    codec = cfds.Codec.CNOCACBP()

    ampal_1ubq = ampal.load_pdb(str(test_file))
    ampal_1ubq2 = ampal.load_pdb(str(test_file))

    test_frame = cfds.voxelise_assembly(
        ampal_1ubq2,
        name="1ubq",
        atom_filter_fn=default_atom_filter,
        frame_edge_length=frame_edge_length,
        voxels_per_side=voxels_per_side,
        encode_cb=True,
        codec=codec,
        tag_rotamers=False,
        chain_dict={},
        voxels_as_gaussian=True,
        verbosity=1,
        chain_filter_list=["A"],
    )[1]

    array_test = []
    for k in range(76):
        array_test.append(test_frame["A"][k].data)

    for atom in ampal_1ubq.get_atoms():
        if not cfds.default_atom_filter(atom):
            del atom.parent.atoms[atom.res_label]
            del atom
    with tempfile.TemporaryDirectory() as tmpdir:
        # Obtain atom encoder:
        output_file_path = cfds.make_frame_dataset(
            structure_files=[test_file],
            output_folder=tmpdir,
            name="test_dataset",
            frame_edge_length=frame_edge_length,
            voxels_per_side=voxels_per_side,
            verbosity=1,
            require_confirmation=False,
            codec=codec,
            voxels_as_gaussian=True,
        )
        with h5py.File(output_file_path, "r") as dataset:
            for n in range(1, 77):
                # check that the frame for all the data frames match between the input
                # arrays and the ones that come out of the HDF5 data set
                residue_number = str(n)
                residue_test = array_test[n - 1]
                hdf5_array = dataset["1ubq"]["A"][residue_number][()]
                npt.assert_array_equal(
                    hdf5_array,
                    residue_test,
                    err_msg=(
                        "The frame in the HDF5 data set should be the same as the "
                        "input frame."
                    ),
                )
            if (
                ampal_1ubq["A"][residue_number].mol_letter
                in standard_amino_acids.keys()
            ):
                polarity = (
                    -1
                    if polarity_Zimmerman[ampal_1ubq["A"][residue_number].mol_letter]
                    < 20
                    else 1
                )
            else:
                polarity = 0
            if polarity == 1:
                assert np.max(residue_test[:, :, :, 5]) > 0
            if polarity == 0:
                assert np.min(residue_test[:, :, :, 5]) < 0


@settings(deadline=700)
@given(integers(min_value=0, max_value=214))
def test_default_atom_filter(residue_number: int):
    assembly = ampal.load_pdb(str(TEST_DATA_DIR / "3qy1.pdb"))
    focus_residue = assembly[0][residue_number]
    backbone_atoms = ("N", "CA", "C", "O")

    for atom in focus_residue:
        filtered_atom = True if atom.res_label in backbone_atoms else False
        filtered_scenario = cfds.default_atom_filter(atom)
        assert (
            filtered_atom == filtered_scenario
        ), f"Expected {atom.res_label} to return {filtered_atom} after filter"


@settings(deadline=700)
@given(integers(min_value=0, max_value=214))
def test_cb_atom_filter(residue_number: int):
    assembly = ampal.load_pdb(str(TEST_DATA_DIR / "3qy1.pdb"))
    focus_residue = assembly[0][residue_number]
    backbone_atoms = ("N", "CA", "C", "O", "CB")

    for atom in focus_residue:
        filtered_atom = True if atom.res_label in backbone_atoms else False
        filtered_scenario = cfds.keep_sidechain_cb_atom_filter(atom)
        assert (
            filtered_atom == filtered_scenario
        ), f"Expected {atom.res_label} to return {filtered_atom} after filter"


def test_add_gaussian_at_position():
    main_matrix = np.zeros((5, 5, 5, 5), dtype=np.float)
    modifiers_triple = (0, 0, 0)
    codec = cfds.Codec.CNOCACB()

    secondary_matrix, atom_idx = codec.encode_gaussian_atom("C", modifiers_triple)
    atom_coord = (1, 1, 1)

    added_matrix = cfds.add_gaussian_at_position(
        main_matrix, secondary_matrix[:, :, :, atom_idx], atom_coord, atom_idx
    )
    # Check general sum:
    np.testing.assert_array_almost_equal(np.sum(added_matrix), 1.0, decimal=2)
    # Check center:
    assert (
        0 < added_matrix[1, 1, 1][0] < 1
    ), f"The central atom should be 1 but got {main_matrix[1, 1, 1, 0]}."
    # Check middle points (in each direction so 6 total points):
    # +---+---+---+
    # | _ | X | _ |
    # | X | 0 | X |
    # | _ | X | _ |
    # +---+---+---+
    # Where 0 is the central atom
    np.testing.assert_array_almost_equal(
        added_matrix[1, 0, 1, 0],
        added_matrix[0, 1, 1, 0],
        decimal=2,
        err_msg=f"The atom should be {added_matrix[0, 1, 1, 0]} but got {main_matrix[1, 0, 1, 0]}.",
    )
    np.testing.assert_array_almost_equal(
        added_matrix[1, 1, 0, 0],
        added_matrix[0, 1, 1, 0],
        decimal=2,
        err_msg=f"The atom should be {added_matrix[0, 1, 1, 0]} but got {main_matrix[1, 1, 0, 0]}.",
    )
    np.testing.assert_array_almost_equal(
        added_matrix[1, 1, 2, 0],
        added_matrix[0, 1, 1, 0],
        decimal=2,
        err_msg=f"The atom should be {added_matrix[0, 1, 1, 0]} but got {main_matrix[1, 1, 2, 0]}.",
    )
    np.testing.assert_array_almost_equal(
        added_matrix[1, 2, 1, 0],
        added_matrix[0, 1, 1, 0],
        decimal=2,
        err_msg=f"The atom should be {added_matrix[0, 1, 1, 0]} but got {main_matrix[1, 2, 1, 0]}.",
    )
    np.testing.assert_array_almost_equal(
        added_matrix[2, 1, 1, 0],
        added_matrix[0, 1, 1, 0],
        decimal=2,
        err_msg=f"The atom should be {added_matrix[0, 1, 1, 0]} but got {main_matrix[2, 1, 1, 0]}.",
    )
    # Check inner corners (in each direction so 12 total points):
    # +---+---+---+
    # | X | _ | X |
    # | _ | 0 | _ |
    # | X | _ | X |
    # +---+---+---+
    np.testing.assert_array_almost_equal(
        added_matrix[0, 1, 0, 0],
        added_matrix[0, 0, 1, 0],
        decimal=4,
        err_msg=f"The atom should be {added_matrix[0, 0, 1, 0]} but got {added_matrix[0, 1, 0, 0]}.",
    )
    np.testing.assert_array_almost_equal(
        added_matrix[0, 1, 2, 0],
        added_matrix[0, 0, 1, 0],
        decimal=4,
        err_msg=f"The atom should be {added_matrix[0, 0, 1, 0]} but got {added_matrix[0, 1, 2, 0]}.",
    )
    np.testing.assert_array_almost_equal(
        added_matrix[0, 2, 1, 0],
        added_matrix[0, 0, 1, 0],
        decimal=4,
        err_msg=f"The atom should be {added_matrix[0, 0, 1, 0]} but got {added_matrix[0, 2, 1, 0]}.",
    )
    np.testing.assert_array_almost_equal(
        added_matrix[1, 0, 0, 0],
        added_matrix[0, 0, 1, 0],
        decimal=4,
        err_msg=f"The atom should be {added_matrix[0, 0, 1, 0]} but got {added_matrix[1, 0, 0, 0]}.",
    )
    np.testing.assert_array_almost_equal(
        added_matrix[1, 0, 2, 0],
        added_matrix[0, 0, 1, 0],
        decimal=4,
        err_msg=f"The atom should be {added_matrix[0, 0, 1, 0]} but got {added_matrix[1, 0, 2, 0]}.",
    )
    np.testing.assert_array_almost_equal(
        added_matrix[1, 2, 0, 0],
        added_matrix[0, 0, 1, 0],
        decimal=4,
        err_msg=f"The atom should be {added_matrix[0, 0, 1, 0]} but got {added_matrix[1, 2, 0, 0]}.",
    )
    np.testing.assert_array_almost_equal(
        added_matrix[1, 2, 2, 0],
        added_matrix[0, 0, 1, 0],
        decimal=4,
        err_msg=f"The atom should be {added_matrix[0, 0, 1, 0]} but got {added_matrix[1, 2, 2, 0]}.",
    )
    np.testing.assert_array_almost_equal(
        added_matrix[2, 0, 1, 0],
        added_matrix[0, 0, 1, 0],
        decimal=4,
        err_msg=f"The atom should be {added_matrix[0, 0, 1, 0]} but got {added_matrix[2, 0, 1, 0]}.",
    )
    np.testing.assert_array_almost_equal(
        added_matrix[2, 1, 0, 0],
        added_matrix[0, 0, 1, 0],
        decimal=4,
        err_msg=f"The atom should be {added_matrix[0, 0, 1, 0]} but got {added_matrix[2, 1, 0, 0]}.",
    )
    np.testing.assert_array_almost_equal(
        added_matrix[2, 1, 2, 0],
        added_matrix[0, 0, 1, 0],
        decimal=4,
        err_msg=f"The atom should be {added_matrix[0, 0, 1, 0]} but got {added_matrix[2, 1, 2, 0]}.",
    )
    np.testing.assert_array_almost_equal(
        added_matrix[2, 2, 1, 0],
        added_matrix[0, 0, 1, 0],
        decimal=4,
        err_msg=f"The atom should be {added_matrix[0, 0, 1, 0]} but got {added_matrix[2, 2, 1, 0]}.",
    )
    # Check outer corners(in each direction so 8 total points):
    # +---+---+---+
    # | X | _ | X |
    # | _ | _ | _ |
    # | X | _ | X |
    # +---+---+---+
    np.testing.assert_array_almost_equal(
        added_matrix[0, 2, 0, 0],
        added_matrix[0, 0, 2, 0],
        decimal=4,
        err_msg=f"The atom should be {added_matrix[0, 0, 2, 0]} but got {added_matrix[0, 2, 0, 0]}.",
    )
    np.testing.assert_array_almost_equal(
        added_matrix[0, 2, 2, 0],
        added_matrix[0, 0, 2, 0],
        decimal=4,
        err_msg=f"The atom should be {added_matrix[0, 0, 2, 0]} but got {added_matrix[0, 2, 2, 0]}.",
    )
    np.testing.assert_array_almost_equal(
        added_matrix[2, 0, 0, 0],
        added_matrix[0, 0, 2, 0],
        decimal=4,
        err_msg=f"The atom should be {added_matrix[0, 0, 2, 0]} but got {added_matrix[2, 0, 0, 0]}.",
    )
    np.testing.assert_array_almost_equal(
        added_matrix[2, 0, 2, 0],
        added_matrix[0, 0, 2, 0],
        decimal=4,
        err_msg=f"The atom should be {added_matrix[0, 0, 2, 0]} but got {added_matrix[2, 0, 2, 0]}.",
    )
    np.testing.assert_array_almost_equal(
        added_matrix[2, 2, 0, 0],
        added_matrix[0, 0, 2, 0],
        decimal=4,
        err_msg=f"The atom should be {added_matrix[0, 0, 2, 0]} but got {added_matrix[2, 2, 0, 0]}.",
    )
    np.testing.assert_array_almost_equal(
        added_matrix[2, 2, 2, 0],
        added_matrix[0, 0, 2, 0],
        decimal=4,
        err_msg=f"The atom should be {added_matrix[0, 0, 2, 0]} but got {added_matrix[2, 2, 2, 0]}.",
    )
    # Add additional point and check whether the sum is 2:
    atom_coord = (2, 2, 2)
    added_matrix = cfds.add_gaussian_at_position(
        added_matrix, secondary_matrix[:, :, :, atom_idx], atom_coord, atom_idx
    )
    np.testing.assert_array_almost_equal(np.sum(added_matrix), 2.0, decimal=2)
    # Add point in top left corner and check whether the normalization still adds up to 1:
    # +---+---+---+
    # | _ | _ | _ |
    # | _ | 0 | X |
    # | _ | X | X |
    # +---+---+---+
    # We are keeping all the X and 0
    atom_coord = (0, 0, 0)
    added_matrix = cfds.add_gaussian_at_position(
        main_matrix, secondary_matrix[:, :, :, atom_idx], atom_coord, atom_idx
    )
    np.testing.assert_array_almost_equal(np.sum(added_matrix), 3.0, decimal=2)
    np.testing.assert_array_less(added_matrix[0, 0, 0][0], 1)
    assert (
        0 < added_matrix[0, 0, 0][0] <= 1
    ), f"The central atom value should be between 0 and 1 but was {added_matrix[0, 0, 0][0]}"
    # Testing N, O, Ca, Cb atom channels. Adding atoms at (0, 0, 0) in different channels:
    N_secondary_matrix, N_atom_idx = codec.encode_gaussian_atom("N", modifiers_triple)
    added_matrix = cfds.add_gaussian_at_position(
        main_matrix, N_secondary_matrix[:, :, :, N_atom_idx], atom_coord, N_atom_idx
    )
    np.testing.assert_array_almost_equal(np.sum(added_matrix), 4.0, decimal=2)
    np.testing.assert_array_less(added_matrix[0, 0, 0][N_atom_idx], 1)
    assert (
        0 < added_matrix[0, 0, 0][N_atom_idx] <= 1
    ), f"The central atom value should be between 0 and 1 but was {added_matrix[0, 0, 0][N_atom_idx]}"
    O_secondary_matrix, O_atom_idx = codec.encode_gaussian_atom("O", modifiers_triple)
    added_matrix = cfds.add_gaussian_at_position(
        main_matrix, O_secondary_matrix[:, :, :, O_atom_idx], atom_coord, O_atom_idx
    )
    np.testing.assert_array_almost_equal(np.sum(added_matrix), 5.0, decimal=2)
    np.testing.assert_array_less(added_matrix[0, 0, 0][O_atom_idx], 1)
    assert (
        0 < added_matrix[0, 0, 0][O_atom_idx] <= 1
    ), f"The central atom value should be between 0 and 1 but was {added_matrix[0, 0, 0][O_atom_idx]}"
    CA_secondary_matrix, CA_atom_idx = codec.encode_gaussian_atom(
        "CA", modifiers_triple
    )
    added_matrix = cfds.add_gaussian_at_position(
        main_matrix, CA_secondary_matrix[:, :, :, CA_atom_idx], atom_coord, CA_atom_idx
    )
    np.testing.assert_array_almost_equal(np.sum(added_matrix), 6.0, decimal=2)
    np.testing.assert_array_less(added_matrix[0, 0, 0][CA_atom_idx], 1)
    assert (
        0 < added_matrix[0, 0, 0][CA_atom_idx] <= 1
    ), f"The central atom value should be between 0 and 1 but was {added_matrix[0, 0, 0][CA_atom_idx]}"
    CB_secondary_matrix, CB_atom_idx = codec.encode_gaussian_atom(
        "CB", modifiers_triple
    )
    added_matrix = cfds.add_gaussian_at_position(
        main_matrix, CB_secondary_matrix[:, :, :, CB_atom_idx], atom_coord, CB_atom_idx
    )
    np.testing.assert_array_almost_equal(np.sum(added_matrix), 7.0, decimal=2)
    np.testing.assert_array_less(added_matrix[0, 0, 0][CB_atom_idx], 1)
    assert (
        0 < added_matrix[0, 0, 0][CB_atom_idx] <= 1
    ), f"The central atom value should be between 0 and 1 but was {CB_atom_idx[0, 0, 0][CA_atom_idx]}"


def test_download_pdb_from_csv_file():
    download_csv = Path("tests/testing_files/csv_pdb_list/pdb_to_test.csv")
    test_file_paths = cfds.download_pdb_from_csv_file(
        download_csv,
        verbosity=1,
        pdb_outpath=TEST_DATA_DIR,
        workers=3,
        voxelise_all_states=False,
    )
    assert (
        TEST_DATA_DIR / "1qys.pdb1" in test_file_paths
    ), f"Expected to find {TEST_DATA_DIR / '1qys.pdb1'} as part of the generated paths."
    assert (
        TEST_DATA_DIR / "3qy1A.pdb1" in test_file_paths
    ), f"Expected to find {TEST_DATA_DIR / '3qy1A.pdb1'} as part of the generated paths."
    assert (
        TEST_DATA_DIR / "6ct4.pdb1" in test_file_paths
    ), f"Expected to find {TEST_DATA_DIR / '6ct4.pdb1'} as part of the generated paths."
    assert (
        TEST_DATA_DIR / "1qys.pdb1"
    ).exists(), f"Expected download of 1QYS to return PDB file"
    assert (
        TEST_DATA_DIR / "3qy1A.pdb1"
    ).exists(), f"Expected download of 3QYA to return PDB file"
    assert (
        TEST_DATA_DIR / "6ct4.pdb1"
    ).exists(), f"Expected download of 6CT4 to return PDB file"
    # Delete files:
    (TEST_DATA_DIR / "1qys.pdb1").unlink(), (TEST_DATA_DIR / "3qy1A.pdb1").unlink(), (
        TEST_DATA_DIR / "6ct4.pdb1"
    ).unlink()
    test_file_paths = cfds.download_pdb_from_csv_file(
        download_csv,
        verbosity=1,
        pdb_outpath=TEST_DATA_DIR,
        workers=3,
        voxelise_all_states=True,
    )
    assert (
        TEST_DATA_DIR / "1qys.pdb"
    ).exists(), f"Expected download of 1QYS to return PDB file"
    assert (
        TEST_DATA_DIR / "3qy1A.pdb"
    ).exists(), f"Expected download of 3QYA to return PDB file"
    (TEST_DATA_DIR / "1qys.pdb").unlink(), (TEST_DATA_DIR / "3qy1A.pdb").unlink()

    for i in range(0, 10):
        pdb_code = f"6ct4_{i}.pdb"
        new_paths = TEST_DATA_DIR / pdb_code
        assert new_paths.exists(), f"Could not find path {new_paths} for {pdb_code}"
        new_paths.unlink()


def test_filter_structures_by_blacklist():
    blacklist_file = Path("tests/testing_files/filter/pdb_to_filter.csv")
    structure_files = []
    for pdb in ["1qys.pdb1", "3qy1A.pdb1", "6ct4.pdb1"]:
        structure_files.append(Path(pdb))
    filtered_structures = cfds.filter_structures_by_blacklist(
        structure_files, blacklist_file
    )
    assert len(structure_files) == 3, f"Expected 3 structures to be in the list"
    assert (
        len(filtered_structures) == 2
    ), f"Expected 2 structures to be in the filtered list"
    assert Path("1qys.pdb1") in filtered_structures, f"Expected 1qys to be in the list"
    assert Path("6ct4.pdb1") in filtered_structures, f"Expected 6CT4 to be in the list"
    assert (
        Path("3qy1A.pdb1") not in filtered_structures
    ), f"Expected 3qy1A not to be in the list"
