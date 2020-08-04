"""Tools for creating a frame dataset.

In this type of dataset, all individual entries are stored separately in a flat
structure.
"""

from dataclasses import dataclass
import glob
import gzip
import multiprocessing as mp
import pathlib
import sys
import time
import typing as t
import warnings

import ampal
import ampal.geometry as geometry
import click
import h5py
import numpy as np

from ampal.amino_acids import standard_amino_acids
from aposteriori.dnn.config import UNCOMMON_RESIDUE_DICT


# {{{ Types
@dataclass
class ResidueResult:
    residue_id: str
    label: str
    encoded_residue: np.ndarray
    data: np.ndarray


@dataclass
class DatasetMetadata:
    atom_encoder: str
    encode_cb: bool
    encoder_length: int
    atom_filter_fn: str
    residue_encoder: t.List[str]


StrOrPath = t.Union[str, pathlib.Path]
ChainDict = t.Dict[str, t.List[ResidueResult]]
# }}}


# {{{ Residue Frame Creation
def align_to_residue_plane(residue: ampal.Residue):
    """Reorients the parent ampal.Assembly that the peptide plane lies on xy.

    Notes
    -----
    This changes the assembly **in place**.

    Parameters
    ----------
    residue: ampal.Residue
        Residue that will be used as a reference to reorient the assemble.
        The assembly will be reoriented so that the residue['CA'] lies on the origin,
        residue['N'] lies on +Y and residue['C'] lies on +X-Y, assuming correct
        geometry.
    """
    # unit vectors used for alignment
    origin = (0, 0, 0)
    unit_y = (0, 1, 0)
    unit_x = (1, 0, 0)

    # translate the whole parent assembly so that residue['CA'] lies on the origin
    translation_vector = residue["CA"].array
    assembly = residue.parent.parent
    assembly.translate(-translation_vector)

    # rotate whole assembly so that N-CA lies on Y
    n_vector = residue["N"].array
    rotation_axis = np.cross(n_vector, unit_y)
    try:
        assembly.rotate(geometry.angle_between_vectors(n_vector, unit_y), rotation_axis)
    except ZeroDivisionError:
        pass

    # align C with xy plane
    rotation_angle = geometry.dihedral(unit_x, origin, unit_y, residue["C"])
    assembly.rotate(-rotation_angle, unit_y)

    return


def encode_cb_to_ampal_residue(residue: ampal.Residue):
    """
    TODO:
    This was calculated by averaging the coordinates of the aligned frames for the 1QYS protein

    Parameters
    ----------
    residue

    Returns
    -------

    """
    avg_cb_position = (-0.741287356, -0.53937931, -1.224287356)
    cb_atom = ampal.base_ampal.Atom(
        avg_cb_position, element="C", res_label="CB", parent=residue
    )
    residue["CB"] = cb_atom
    return


def within_frame(frame_edge_length: float, atom: ampal.Atom) -> bool:
    """Tests if an atom is within the `frame_edge_length` of the origin."""
    half_frame_edge_length = frame_edge_length / 2
    return all([0 <= abs(v) <= half_frame_edge_length for v in atom.array])


def discretize(
    atom: ampal.Atom, voxel_edge_length: float, adjust_by: int = 0
) -> t.Tuple[int, int, int]:
    """Rounds and then converts to an integer.

    Parameters
    ----------
    atom: ampal.Atom
        Atom x, y, z coordinates will be discretized based on `voxel_edge_length`.
    voxel_edge_length: float
        Edge length of the voxels that are mapped onto cartesian space.
    adjust_by: int
    """

    # I'm explicitly repeating this operation to make it explicit to the type checker
    # that a triple is returned.
    return (
        int(np.round(atom.x / voxel_edge_length)) + adjust_by,
        int(np.round(atom.y / voxel_edge_length)) + adjust_by,
        int(np.round(atom.z / voxel_edge_length)) + adjust_by,
    )


def encode_atom(atom: ampal.Atom, encoder_length: int) -> np.ndarray:
    """
    Encodes atoms in a boolean array depending on the type of encoding chosen.

    Parameters
    ----------
    atom: ampal.Atom
        Atom to be encoded.
    encoder_length: int
        Length of the number of channels needed in the encoded atoms.

    Returns
    -------
    atom_encoding: np.ndarray
        Boolean array with atom encoding of shape (encoder_length,)

    """
    atom_label = atom.res_label
    # Creating empty atom encoding
    atom_encoding = np.zeros(encoder_length, dtype=bool)
    # Attempt encoding
    if atom_label == "C":
        atom_encoding[0] = True
    elif atom_label == "N":
        atom_encoding[1] = True
    elif atom_label == "O":
        atom_encoding[2] = True
    elif atom_label == "CB":
        atom_encoding[3] = True
    elif atom_label == "CA":
        if encoder_length == 3 or 4:
            # Add General C encoding
            atom_encoding[0] = True
        elif encoder_length == 5:
            atom_encoding[4] = True
        else:
            assert encoder_length in [
                3,
                4,
                5,
            ], f"Expected CNO (3), CNOCB (4), CNOCACB (5) encoding (classes) but got encoding with {encoder_length} classes."
    else:
        warnings.warn(
            f"Attempted encoding of atom {atom_label}, but it is not C, N, O, CB or CA. Returning 0 instead."
        )

    return atom_encoding


def encode_residue(residue: str) -> np.ndarray:
    """
    TODO:
    Parameters
    ----------
    residue

    Returns
    -------

    """
    std_residues = list(standard_amino_acids.values())
    residue_encoding = np.zeros(len(std_residues), dtype=bool)

    # Deal with non-standard residues:
    if residue not in std_residues:
        if residue in UNCOMMON_RESIDUE_DICT.keys():
            warnings.warn(f"{residue} is not a standard residue.")
            residue_label = UNCOMMON_RESIDUE_DICT[residue]
            warnings.warn(f"Residue converted to {residue}.")
        else:
            assert (
                residue in UNCOMMON_RESIDUE_DICT.keys()
            ), f"Expected natural amino acid, attempted conversion from uncommon residues, but got {residue}."
    else:
        residue_label = residue

    # Add True at the correct residue index:
    res_idx = std_residues.index(residue_label)
    residue_encoding[res_idx] = 1

    return residue_encoding


def create_residue_frame(
    residue: ampal.Residue,
    frame_edge_length: float,
    voxels_per_side: int,
    encode_cb: bool,
    atom_encoder: str,
    encoder_length: int,
) -> np.ndarray:
    """Creates a discrete representation of a volume of space around a residue.

    Notes
    -----
    We use the term "frame" to refer to a cube of space around a residue.

    Parameters
    ----------
    residue: ampal.Residue
        The residue to be converted to a frame.
    frame_edge_length: float
        The length of the edges of the frame.
    voxels_per_side: int
        The number of voxels per edge that the cube of space will be converted into i.e.
        the final cube will be `voxels_per_side`^3. This must be a odd, positive integer
        so that the CA atom can be placed at the centre of the frame.
    encode_cb: bool
        Whether to encode the Cb at an average position in the frame.
    atom_encoder: str
        Whether to encode the frames as C, N, O or have additional channels for Ca and Cb.
        Options available are: "CNO", "CNOCB", "CNOCBCA".
    encoder_length: int
        Length of the number of channels needed in the encoded atoms.

    Returns
    -------
    frame: ndarray
        Numpy array containing the discrete representation of a cube of space around the
        residue.
    
    Raises
    ------
    AssertionError
        Raised if:
        
        * If any atom does not have an element label.
        * If any residue does not have a three letter `mol_code` i.e. "LYS" etc
        * If any voxel is already occupied
        * If the central voxel in the frame is not carbon as it should the the CA atom
    """
    assert voxels_per_side % 2, "The number of voxels per side should be odd."
    voxel_edge_length = frame_edge_length / voxels_per_side
    assembly = residue.parent.parent
    chain = residue.parent

    align_to_residue_plane(residue)
    # Create a Cb atom at avg postion:
    if atom_encoder in ["CNOCB", "CNOCBCA"]:
        if encode_cb:
            encode_cb_to_ampal_residue(residue)

    frame = np.zeros(
        (voxels_per_side, voxels_per_side, voxels_per_side, encoder_length), dtype=bool
    )
    # iterate through all atoms within the frame
    for atom in (
        a
        for a in assembly.get_atoms(ligands=False)
        if within_frame(frame_edge_length, a)
    ):
        # 3d coordinates are converted to relative indices in frame array
        indices = discretize(atom, voxel_edge_length, adjust_by=voxels_per_side // 2)
        ass = atom.parent.parent.parent
        cha = atom.parent.parent
        res = atom.parent
        assert (atom.element != "") or (atom.element != " "), (
            f"Atom element should not be blank:\n"
            f"{atom.chain}:{atom.res_num}:{atom.res_label}"
        )
        assert (res.mol_code != "") or (res.mol_code != " "), (
            f"Residue mol_code should not be blank:\n"
            f"{cha.id}:{res.id}:{atom.res_label}"
        )
        np.testing.assert_array_equal(
            frame[indices], np.array([False]*len(frame[indices]), dtype=bool)
        )
        assert frame[indices][0] == False, (
            f"Voxel should not be occupied: Currently "
            f"{frame[indices]}, "
            f"{ass.id}:{cha.id}:{res.id}:{atom.res_label}"
        )
        frame[indices] = encode_atom(atom, encoder_length)
    centre = voxels_per_side // 2
    if atom_encoder in ["CNO", "CNOCB"]:
        assert (
            frame[centre, centre, centre][0] == 1
        ), f"The central atom should be Carbon, but it is {frame[centre, centre, centre]}."
    elif atom_encoder == "CNOCBCA":
        assert (
            frame[centre, centre, centre][4] == 1
        ), f"The central atom should be Carbon, but it is {frame[centre, centre, centre]}."
    return frame


def create_frames_from_structure(
    structure_path: pathlib.Path,
    frame_edge_length: float,
    voxels_per_side: int,
    atom_filter_fn: t.Callable[[ampal.Atom], bool],
    chain_filter_list: t.Optional[t.List[str]],
    gzipped: bool,
    verbosity: int,
    encode_cb: bool,
    atom_encoder: str,
    encoder_length: int,
) -> t.Tuple[str, ChainDict]:
    """Creates residue frames for each residue in the structure.

    Parameters
    ----------
    structure_path: pathlib.Path
        Path to pdb file to be processed into frames
    frame_edge_length: float
        The length of the edges of the frame.
    voxels_per_side: int
        The number of voxels per edge that the cube of space will be converted into i.e.
        the final cube will be `voxels_per_side`^3. This must be a odd, positive integer
        so that the CA atom can be placed at the centre of the frame.
    atom_filter_fn: ampal.Atom -> bool
        A function used to preprocess structures to remove atoms that are not to be
        included in the final structure. By default water and side chain atoms will be
        removed.
    chain_filter_list: t.Optional[t.List[str]]
        Chains to be processed.
    gzipped: bool
        Indicates if structure files are gzipped or not.
    verbosity: int
        Level of logging sent to std out.
    encode_cb: bool
        Whether to encode the Cb at an average position in the frame. If
        True, it will not be filtered by the `atom_filter_fn`.
    atom_encoder: str
        Whether to encode the frames as C, N, O or have additional channels for Ca and Cb.
        Options available are: "CNO", "CNOCB", "CNOCBCA".
    encoder_length: int
        Length of the number of channels needed in the encoded atoms.

    """
    name = structure_path.name.split(".")[0]
    chain_dict: ChainDict = {}
    if gzipped:
        with gzip.open(str(structure_path), "rb") as inf:
            assembly = ampal.load_pdb(inf.read().decode(), path=False)[0]
    else:
        assembly = ampal.load_pdb(str(structure_path))
    # Deals with structures from NMR as ampal returns Container of Assemblies
    if isinstance(assembly, ampal.AmpalContainer):
        warnings.warn(f"Selecting the first state from the NMR structure {assembly.id}")
        assembly = assembly[0]
    # Filters atoms not related to assembly:
    total_atoms = len(list(assembly.get_atoms()))
    for atom in assembly.get_atoms():
        if not atom_filter_fn(atom):
            del atom.parent.atoms[atom.res_label]
            del atom
    remaining_atoms = len(list(assembly.get_atoms()))
    print(f"{name}: Filtered {total_atoms-remaining_atoms} of {total_atoms} atoms.")
    for chain in assembly:
        if chain_filter_list:
            if chain.id.upper() not in chain_filter_list:
                if verbosity > 0:
                    print(
                        f"{name}:\tIgnoring chain {chain.id}, not in Pieces filter "
                        f"file."
                    )
                continue
        if not isinstance(chain, ampal.Polypeptide):
            if verbosity > 0:
                print(f"{name}:\tIgnoring non-polypeptide chain ({chain.id}).")
            continue
        if verbosity > 0:
            print(f"{name}:\tProcessing chain {chain.id}...")
        chain_dict[chain.id] = []
        # Loop through each residue, voxelis:
        for residue in chain:
            if isinstance(residue, ampal.Residue):
                # Create voxelised frame:
                array = create_residue_frame(
                    residue,
                    frame_edge_length,
                    voxels_per_side,
                    encode_cb,
                    atom_encoder,
                    encoder_length,
                )
                encoded_residue = encode_residue(residue.mol_code)
                # Save results:
                chain_dict[chain.id].append(
                    ResidueResult(
                        residue_id=str(residue.id),
                        label=residue.mol_code,
                        encoded_residue=encoded_residue,
                        data=array,
                    )
                )
                if verbosity > 1:
                    print(f"{name}:\t\tAdded residue {chain.id}:{residue.id}.")
        if verbosity > 0:
            print(f"{name}:\tFinished processing chain {chain.id}.")
    return (name, chain_dict)


# }}}
# {{{ Dataset Creation
def default_atom_filter(atom: ampal.Atom) -> bool:
    """Filters for all heavy protein backbone atoms."""
    backbone_atoms = ("N", "CA", "C", "O")
    if atom.element == "H":
        return False
    elif isinstance(atom.parent, ampal.Residue) and (atom.res_label in backbone_atoms):
        return True
    else:
        return False


def keep_sidechain_cb_atom_filter(atom: ampal.Atom) -> bool:
    """Filters for all heavy protein backbone atoms and the Beta Carbon of
    the side-chain."""
    atoms_to_keep = ("N", "CA", "C", "O", "CB")
    if atom.element == "H":
        return False
    elif isinstance(atom.parent, ampal.Residue) and (atom.res_label in atoms_to_keep):
        return True
    else:
        return False


def process_single_path(
    path_queue: mp.SimpleQueue,
    result_queue: mp.SimpleQueue,
    frame_edge_length: float,
    voxels_per_side: int,
    atom_filter_fn: t.Callable[[ampal.Atom], bool],
    chain_filter_dict: t.Optional[t.Dict[str, t.List[str]]],
    errors: t.Dict[str, str],
    gzipped: bool,
    verbosity: int,
    encode_cb: bool,
    atom_encoder: str,
    encoder_length: int,
):
    """Processes a path and puts the results into a queue."""
    chain_filter_list: t.Optional[t.List[str]]
    result: t.Union[t.Tuple[str, ChainDict], str]
    while True:
        structure_path = path_queue.get()
        print(f"Processing `{structure_path}`...")
        try:
            if chain_filter_dict:
                chain_filter_list = chain_filter_dict[
                    structure_path.name.split(".")[0].upper()
                ]
            else:
                chain_filter_list = None
            result = create_frames_from_structure(
                structure_path,
                frame_edge_length,
                voxels_per_side,
                atom_filter_fn,
                chain_filter_list,
                gzipped,
                verbosity,
                encode_cb,
                atom_encoder,
                encoder_length,
            )
        except Exception as e:
            result = str(e)
        if isinstance(result, str):
            errors[str(structure_path)] = result
        else:
            result_queue.put(result)


def save_results(
    result_queue: mp.SimpleQueue,
    h5_path: pathlib.Path,
    total_files: int,
    complete: mp.Value,
    frames: mp.Value,
    verbosity: int,
    metadata: DatasetMetadata,
):
    """Saves voxelized structures to a hdf5 object."""
    with h5py.File(str(h5_path), "w") as hd5:
        while True:
            # Requires explicit type annotation as I can't figure out how to annotate
            # the SimpleQueue object directly
            result: t.Tuple[str, ChainDict] = result_queue.get()
            if result == "BREAK":
                break
            pdb_code, chain_dict = result
            print(f"{pdb_code}: Storing results...")
            if pdb_code in hd5:
                print(f"{pdb_code}:\t\tError PDB already found in dataset skipping.")
            else:
                # Encode metadata:
                metadata_dict = metadata.__dict__
                # Loop through metadata dataclass and add it as attribute:
                for meta in list(metadata_dict.keys()):
                    hd5.attrs[str(meta)] = metadata_dict[meta]

                pdb_group = hd5.create_group(pdb_code)
                for chain_id, res_results in chain_dict.items():
                    if verbosity > 0:
                        print(f"{pdb_code}:\tStoring chain {chain_id}...")
                    if chain_id in pdb_group:
                        print(
                            f"{pdb_code}:\t\tError chain {chain_id} found in dataset, "
                            f"skipping."
                        )
                        continue
                    chain_group = pdb_group.create_group(chain_id)
                    for res_result in res_results:
                        if verbosity > 1:
                            print(
                                f"{pdb_code}:\t\tStoring chain {res_result.residue_id}..."
                            )
                        if res_result.residue_id in chain_group:
                            print(
                                f"{pdb_code}:\t\tError {res_result.residue_id} in "
                                f"chain group, skipping."
                            )
                            continue
                        res_dataset = chain_group.create_dataset(
                            res_result.residue_id, data=res_result.data, dtype=bool,
                        )
                        res_dataset.attrs["label"] = res_result.label
                        res_dataset.attrs[
                            "encoded_residue"
                        ] = res_result.encoded_residue
                        frames.value += 1
                print(f"{pdb_code}: Finished processing.")
            complete.value += 1
            print(f"Files processed {complete.value}/{total_files}.")
        print(f"Finished processing files.")


def process_paths(
    structure_file_paths: t.List[pathlib.Path],
    output_path: pathlib.Path,
    frame_edge_length: float,
    voxels_per_side: int,
    atom_filter_fn: t.Callable[[ampal.Atom], bool],
    chain_filter_dict: t.Optional[t.Dict[str, t.List[str]]],
    processes: int,
    gzipped: bool,
    verbosity: int,
    encode_cb: bool,
    atom_encoder: str,
):
    """Discretizes a list of structures and stores them in a HDF5 object.

    Parameters
    ----------
    structure_file_paths: List[pathlib.Path]
        List of paths to pdb files to be processed into frames
    output_path: pathlib.Path
        Path where dataset will be written.
    frame_edge_length: float
        The length of the edges of the frame.
    voxels_per_side: int
        The number of voxels per edge that the cube of space will be converted into i.e.
        the final cube will be `voxels_per_side`^3. This must be a odd, positive integer
        so that the CA atom can be placed at the centre of the frame.
    atom_filter_fn: ampal.Atom -> bool
        A function used to preprocess structures to remove atoms that are not to be
        included in the final structure. By default water and side chain atoms will be
        removed.
    chain_filter_dict: t.Optional[t.Dict[str, t.List[str]]]
        Chains to be selected from the PDB file.
    processes: int
        Number of processes to used to process structure files.
    gzipped: bool
        Indicates if structure files are gzipped or not.
    verbosity: int
        Level of logging sent to std out.
    encode_cb: bool
        Whether to encode the Cb at an average position in the frame.
    atom_encoder: str
        Whether to encode the frames as C, N, O or have additional channels for Ca and Cb.
        Options available are: "CNO", "CNOCB", "CNOCBCA".
    """

    with mp.Manager() as manager:
        # Need to ignore the type here due to a weird problem with the Queue type not
        # being found
        path_queue = manager.Queue()  # type: ignore
        total_paths = len(structure_file_paths)
        for path in structure_file_paths:
            path_queue.put(path)
        result_queue = manager.Queue()  # type: ignore
        complete = manager.Value("i", 0)  # type: ignore
        frames = manager.Value("i", 0)  # type: ignore
        errors = manager.dict()  # type: ignore
        total = len(structure_file_paths)
        # create an empty array for discrete frame
        if atom_encoder == "CNO":
            encoder_length = 3
        elif atom_encoder == "CNOCB":
            encoder_length = 4
        elif atom_encoder == "CNOCBCA":
            encoder_length = 5
        else:
            assert atom_encoder in [
                "CNO",
                "CNOCB",
                "CNOCBCA",
            ], f"Expected encoder type to be CNO, CNOCB, CNOCBCA, but got {atom_encoder}."

        workers = [
            mp.Process(
                target=process_single_path,
                args=(
                    path_queue,
                    result_queue,
                    frame_edge_length,
                    voxels_per_side,
                    atom_filter_fn,
                    chain_filter_dict,
                    errors,
                    gzipped,
                    verbosity,
                    encode_cb,
                    atom_encoder,
                    encoder_length,
                ),
            )
            for proc_i in range(processes)
        ]
        metadata = DatasetMetadata(atom_encoder, encode_cb, encoder_length, str(atom_filter_fn), list(standard_amino_acids.values()))
        storer = mp.Process(
            target=save_results,
            args=(
                result_queue,
                output_path,
                total_paths,
                complete,
                frames,
                verbosity,
                metadata,
            ),
        )
        all_processes = workers + [storer]
        for proc in all_processes:
            proc.start()
        while (complete.value + len(errors)) < total:
            if not all([p.is_alive() for p in all_processes]):
                print("One or more of the processes died, aborting...")
                break
            time.sleep(5)
        else:
            result_queue.put("BREAK")
            storer.join()
        for proc in all_processes:
            proc.terminate()
        if (verbosity > 0) and (errors):
            print(f"There were {len(errors)} errors while creating the dataset:")
            for path, error in errors.items():
                print(f"\t{path}:")
                print(f"\t\t{error}")
        else:
            print(f"There were {len(errors)} errors while creating the dataset.")
        print(
            f"Created frame dataset at `{output_path.resolve()}` containing "
            f"{frames.value} residue frames."
        )
    return


def make_frame_dataset(
    structure_files: t.List[StrOrPath],
    output_folder: StrOrPath,
    name: str,
    frame_edge_length: float,
    voxels_per_side: int,
    atom_filter_fn: t.Callable[[ampal.Atom], bool] = default_atom_filter,
    pieces_filter_file: t.Optional[StrOrPath] = None,
    processes: int = 1,
    gzipped: bool = False,
    verbosity: int = 1,
    require_confirmation: bool = True,
    encode_cb: bool = True,
    atom_encoder: str = "CNO",
) -> pathlib.Path:
    """Creates a dataset of voxelized amino acid frames.

    Parameters
    ----------
    structure_files: List[str or pathlib.Path]
        List of paths to pdb files to be processed into frames
    output_folder: StrOrPath
        Path to folder where output will be written.
    name: str
        Name used for the dataset file, `.hd5` will be appended.
    frame_edge_length: float
        The length of the edges of the frame.
    voxels_per_side: int
        The number of voxels per edge that the cube of space will be converted into i.e.
        the final cube will be `voxels_per_side`^3. This must be a odd, positive integer
        so that the CA atom can be placed at the centre of the frame.
    atom_filter_fn: ampal.Atom -> bool
        A function used to preprocess structures to remove atoms that are not to be
        included in the final structure. By default water and side chain atoms will be
        removed.
    pieces_filter_file: Optional[StrOrPath]
        A path to a Pieces file that will be used to filter the input files and specify
        chains to be included in the dataset.
    processes: int
        Number of processes to used to process structure files.
    gzipped: bool
        Indicates if structure files are gzipped or not.
    verbosity: int
        Level of logging sent to std out.
    require_confirmation: bool
        If True, the user will be prompted to start creating the dataset.
    encode_cb: bool
        Whether to encode the Cb at an average position in the frame.
    atom_encoder: str
        Whether to encode the frames as C, N, O or have additional channels for Ca and Cb.
        Options available are: "CNO", "CNOCB", "CNOCBCA".
    Returns
    -------
    output_file_path: pathlib.Path
        A path to the location of the output dataset.
    """
    chain_filter_dict: t.Optional[t.Dict[str, t.List[str]]]

    assert len(structure_files) > 0, "Aborting, no structure files defined."
    assert (
        voxels_per_side % 2
    ), "`voxels-per-side` must be odd, so that the CA is centred."
    if pieces_filter_file:
        # Assuming standard Pieces format, ignore first row, take first column and
        # split into PDB code and chain
        with open(pieces_filter_file, "r") as inf:
            chain_filter_dict = {}
            _ = inf.__next__()
            for line in inf:
                pdb_code = line[:4]
                chain_id = line[4]
                if not pdb_code in chain_filter_dict:
                    chain_filter_dict[pdb_code] = []
                chain_filter_dict[pdb_code].append(chain_id)
    else:
        chain_filter_dict = None
    structure_file_paths = [pathlib.Path(x) for x in structure_files]
    if chain_filter_dict:
        original_path_num = len(structure_file_paths)
        structure_file_paths = [
            p
            for p in structure_file_paths
            if p.name.split(".")[0].upper() in chain_filter_dict
        ]
        print(
            f"{original_path_num - len(structure_file_paths)} structure file/s were "
            f"not found in the Pieces filter file, these will not be processed."
        )
    output_file_path = pathlib.Path(output_folder) / (name + ".hdf5")
    total_files = len(structure_file_paths)
    processed_files = 0
    number_of_frames = 0

    print(f"Will attempt to process {total_files} structure file/s.")
    print(f"Output file will be written to `{output_file_path.resolve()}`.")
    voxel_edge_length = frame_edge_length / voxels_per_side
    max_voxel_distance = np.sqrt(voxel_edge_length ** 2 * 3)
    print(f"Frame edge length = {frame_edge_length:.2f} A")
    print(f"Voxels per side = {voxels_per_side}")
    print(f"Voxels will have an edge length of {voxel_edge_length:.2f} A.")
    print(f"Max internal distance of each voxel will be {max_voxel_distance:.2f} A.")
    if require_confirmation:
        print("Do you want to continue? [y]/n")
        response = input()
        if not ((response == "") or (response == "y")):
            print("Aborting.")
            sys.exit()
    process_paths(
        structure_file_paths=structure_file_paths,
        output_path=output_file_path,
        frame_edge_length=frame_edge_length,
        voxels_per_side=voxels_per_side,
        processes=processes,
        atom_filter_fn=atom_filter_fn,
        chain_filter_dict=chain_filter_dict,
        gzipped=gzipped,
        verbosity=verbosity,
        encode_cb=encode_cb,
        atom_encoder=atom_encoder,
    )
    return output_file_path


# }}}
# {{{ CLI
@click.command()
@click.argument(
    "structure_file_folder", type=click.Path(exists=True, readable=True),
)
@click.option(
    "-o",
    "--output-folder",
    type=click.Path(),
    default=".",
    help=("Path to folder where output will be written. Default = `.`"),
)
@click.option(
    "-n",
    "--name",
    type=str,
    default="frame_dataset",
    help=(
        "Name used for the dataset file, the `.hdf5` extension does not need to be "
        "included as it will be appended. Default = `frame_dataset`"
    ),
)
@click.option(
    "-e",
    "--extension",
    type=str,
    default=".pdb",
    help=("Extension of structure files to be included. Default = `.pdb`."),
)
@click.option(
    "--pieces-filter-file",
    type=click.Path(),
    help=(
        "Path to a Pieces format file used to filter the dataset to specific chains in"
        "specific files. All other PDB files included in the input will be ignored."
    ),
)
@click.option(
    "--frame-edge-length",
    type=float,
    default=12.0,
    help=(
        "Edge length of the cube of space around each residue that will be voxelized. "
        "Default = 12.0 Angstroms."
    ),
)
@click.option(
    "--voxels-per-side",
    type=int,
    default=21,
    help=(
        "The number of voxels per side of the frame. This will give a final cube of "
        "`voxels-per-side`^3. Default = 21."
    ),
)
@click.option(
    "-p",
    "--processes",
    type=int,
    default=1,
    help=("Number of processes to be used to create the dataset. Default = 1."),
)
@click.option(
    "-z",
    "--gzipped",
    is_flag=True,
    help=(
        "If True, this flag indicates that the structure files are gzipped. Default = "
        "False."
    ),
)
@click.option(
    "-r",
    "--recursive",
    is_flag=True,
    help=("If True, all files in all subfolders will be processed."),
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help=(
        "Sets the verbosity of the output, use `-v` for low level output or `-vv` for "
        "even more information."
    ),
)
@click.option(
    "-cb",
    "--encode_cb",
    type=bool,
    default=True,
    help=(
        "Encode the Cb at an average position (-0.741287356, -0.53937931, -1.224287356) in the aligned frame, even for Glycine residues."
    ),
)
@click.option(
    "-ae",
    "--atom_encoder",
    type=str,
    default="CNO",
    help=(
        "Encodes atoms in different channels, depending on atom types. Default is CNO, other options are ´CNOCB´ and `CNOCBCA` to encode the Cb or Cb and Ca in different channels respectively."
    ),
)
def cli(
    structure_file_folder: str,
    output_folder: str,
    name: str,
    extension: str,
    pieces_filter_file: str,
    frame_edge_length: float,
    voxels_per_side: int,
    processes: int,
    gzipped: bool,
    recursive: bool,
    verbose: int,
    encode_cb: bool,
    atom_encoder: str,
):
    """Creates a dataset of voxelized amino acid frames.

    A frame refers to a region of space around an amino acid. For every
    residue in the input structure(s), a cube of space around the region
    (with an edge length equal to `--frame_edge_length`, default 12 Å),
    will be mapped to discrete space, with a defined number of voxels per
    edge (equal to `--voxels-per-side`, default = 21).

    Basic Usage:

    `make-frame-dataset -o /tmp/ -n test_dataset 1ubq.pdb 1ctf.pdb`

    This command will make a tiny dataset found at `/tmp/test_dataset.hdf5`,
    containing all residues 1ubq.pdb and 1ctf.pdb

    Globs can be used to define the structure files to be processed.
    `make-frame-dataset pdb_files/**/*.pdb` would include all `.pdb` files in all
    subdirectories of the `pdb_files` directory.

    You can process gzipped pdb files, but the program assumes that the format
    of the file name is similar to `1mkk.pdb.gz`. If you have more complex
    requirements than this, we recommend using this library directly from
    Python rather than through this CLI.

    The hdf5 object itself is like a Python dict. The structure is
    simple:

    hdf5 Contains a number of groups, one for each structure file.
    └─[pdb_code] Contains a number of subgroups, one for each chain.
      └─[chain_id] Contains a number of subgroups, one for each residue.
        └─[residue_id] voxels_per_side^3 array of ints, representing element number.
          └─.attrs['label'] Three-letter code for the residue.

    So hdf5['1ctf']['A']['58'] would be an array for the voxelized.
    """
    structure_folder_path = pathlib.Path(structure_file_folder)
    structure_files: t.List[StrOrPath] = list(
        structure_folder_path.glob(f"**/*{extension}")
        if recursive
        else structure_folder_path.glob(f"*{extension}")
    )
    if not structure_files:
        print(
            f"No structure_files found in `{structure_folder_path}`. Did you mean to "
            f"use the recursive flag?"
        )
        sys.exit()
    make_frame_dataset(
        structure_files,
        output_folder,
        name,
        frame_edge_length,
        voxels_per_side,
        default_atom_filter,
        pieces_filter_file,
        processes,
        gzipped,
        verbose,
        encode_cb=encode_cb,
        atom_encoder=atom_encoder,
    )
    return


# }}}
if __name__ == "__main__":
    # The cli will be run if this file is invoked directly
    # It is also hooked up as a script in `pyproject.toml`
    cli()
