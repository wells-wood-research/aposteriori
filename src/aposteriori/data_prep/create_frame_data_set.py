"""Tools for creating a frame dataset.

In this type of dataset, all individual entries are stored separately in a flat
structure.
"""
import csv
import glob
import gzip
import multiprocessing as mp
import pathlib
import sys
import time
import typing as t
import urllib
import warnings
from dataclasses import dataclass

import ampal
import ampal.geometry as geometry
import h5py
import numpy as np

from ampal.amino_acids import standard_amino_acids
from aposteriori.dnn.config import (
    UNCOMMON_RESIDUE_DICT,
    MAKE_FRAME_DATASET_VER,
    PDB_PATH,
    PDB_REQUEST_URL,
)


# {{{ Types
@dataclass
class ResidueResult:
    residue_id: str
    label: str
    encoded_residue: np.ndarray
    data: np.ndarray


@dataclass
class DatasetMetadata:
    make_frame_dataset_ver: str
    frame_dims: t.Tuple[int, int, int, int]
    atom_encoder: t.List[str]
    encode_cb: bool
    atom_filter_fn: str
    residue_encoder: t.List[str]
    frame_edge_length: float

    @classmethod
    def import_metadata_dict(cls, meta_dict: t.Dict[str, t.Any]):
        """
        Imports metada of a dataset from a dictionary object to the DatasetMetadata class.

        Parameters
        ----------
        meta_dict: t.Dict[str, t.Any]
            Dictionary of metadata parameters for the dataset.

        Returns
        -------
        DatasetMetadata dataclass with filled metadata.

        """
        return cls(**meta_dict)


StrOrPath = t.Union[str, pathlib.Path]
ChainDict = t.Dict[str, t.List[ResidueResult]]
# }}}


# {{{ Residue Frame Creation
class Codec:
    def __init__(self, atomic_labels: t.List[str]):
        # Set attributes:
        self.atomic_labels = atomic_labels
        self.encoder_length = len(self.atomic_labels)
        self.label_to_encoding = dict(
            zip(self.atomic_labels, range(self.encoder_length))
        )
        self.encoding_to_label = dict(
            zip(range(self.encoder_length), self.atomic_labels)
        )
        return

    # Labels Class methods:
    @classmethod
    def CNO(cls):
        return cls(["C", "N", "O"])

    @classmethod
    def CNOCB(cls):
        return cls(["C", "N", "O", "CB"])

    @classmethod
    def CNOCBCA(cls):
        return cls(["C", "N", "O", "CB", "CA"])

    def encode_atom(self, atom_label: str) -> np.ndarray:
        """
        Encodes atoms in a boolean array depending on the type of encoding chosen.

        Parameters
        ----------
        atom_label: str
            Label of the atom to be encoded.

        Returns
        -------
        atom_encoding: np.ndarray
            Boolean array with atom encoding of shape (encoder_length,)

        """
        # Creating empty atom encoding:
        encoded_atom = np.zeros(self.encoder_length, dtype=bool)
        # Attempt encoding:
        if atom_label in list(self.label_to_encoding.keys()):
            atom_idx = self.label_to_encoding[atom_label]
            encoded_atom[atom_idx] = True
        # Encode CA as C in case it is not in the labels:
        elif atom_label == "CA":
            atom_idx = self.label_to_encoding["C"]
            encoded_atom[atom_idx] = True
        else:
            warnings.warn(
                f"{atom_label} not found in {self.atomic_labels} encoding. Returning None."
            )

        return encoded_atom

    def decode_atom(self, encoded_atom: np.ndarray) -> t.Optional[str]:
        """
        Decodes atoms into string depending on the type of encoding chosen.

        Parameters
        ----------
        encoded_atom: np.ndarray
            Boolean array with atom encoding of shape (encoder_length,)

        Returns
        -------
        decoded_atom: t.Optional[str]
            Label of the decoded atom.
        """
        # Get True index of one-hot encoding
        atom_encoding = np.nonzero(encoded_atom)[0]

        if atom_encoding.size == 0:
            warnings.warn(f"Encoded atom was 0.")
        # If not Empty space:
        else:
            # Decode Atom:
            decoded_atom = self.encoding_to_label[atom_encoding[0]]
            return decoded_atom


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
    Encodes a Cb atom to an AMPAL residue. The Cb is added to an average position
    calculated by averaging the Cb coordinates of the aligned frames for the 1QYS protein.

    Parameters
    ----------
    residue: ampal.Residue
        Focus residues that requires the Cb atom.

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


def encode_residue(residue: str) -> np.ndarray:
    """
    One-Hot Encodes a residue string to a numpy array. Attempts to convert non-standard
    residues using AMPAL's UNCOMMON_RESIDUE_DICT.

    Parameters
    ----------
    residue: str
        Residue label of the frame.

    Returns
    -------
    residue_encoding: np.ndarray
        One-Hot encoding of the residue with shape (20,)
    """
    std_residues = list(standard_amino_acids.values())
    residue_encoding = np.zeros(len(std_residues), dtype=bool)

    # Deal with non-standard residues:
    if residue not in std_residues:
        if residue in UNCOMMON_RESIDUE_DICT.keys():
            warnings.warn(f"{residue} is not a standard residue.")
            residue_label = UNCOMMON_RESIDUE_DICT[residue]
            warnings.warn(f"Residue converted to {residue_label}.")
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
    codec: object,
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
    codec: object
        Codec object with encoding instructions.

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
    if "CB" in codec.atomic_labels:
        if encode_cb:
            encode_cb_to_ampal_residue(residue)

    frame = np.zeros(
        (voxels_per_side, voxels_per_side, voxels_per_side, codec.encoder_length),
        dtype=bool,
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
            frame[indices], np.array([False] * len(frame[indices]), dtype=bool)
        )
        assert frame[indices][0] == False, (
            f"Voxel should not be occupied: Currently "
            f"{frame[indices]}, "
            f"{ass.id}:{cha.id}:{res.id}:{atom.res_label}"
        )
        frame[indices] = Codec.encode_atom(codec, atom.res_label)
    centre = voxels_per_side // 2

    # Check whether central atom is C:
    if "CA" in codec.atomic_labels:
        assert (
            frame[centre, centre, centre][4] == 1
        ), f"The central atom should be Carbon, but it is {frame[centre, centre, centre]}."
    else:
        assert (
            frame[centre, centre, centre][0] == 1
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
    codec: object,
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
    codec: object
        Codec object with encoding instructions.

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
                    residue=residue,
                    frame_edge_length=frame_edge_length,
                    voxels_per_side=voxels_per_side,
                    encode_cb=encode_cb,
                    codec=codec,
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
    """Filters all heavy protein backbone atoms."""
    backbone_atoms = ("N", "CA", "C", "O")
    if atom.element == "H":
        return False
    elif isinstance(atom.parent, ampal.Residue) and (atom.res_label in backbone_atoms):
        return True
    else:
        return False


def keep_sidechain_cb_atom_filter(atom: ampal.Atom) -> bool:
    """Filters all heavy protein backbone atoms and the Beta Carbon of
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
    codec: object,
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
                    structure_path.name.split(".")[0].upper().strip("PDB")
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
                codec,
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
                for meta, meta_attribute in metadata_dict.items():
                    hd5.attrs[str(meta)] = meta_attribute

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
                            res_result.residue_id,
                            data=res_result.data,
                            dtype=bool,
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
    codec: object,
):
    """Discretizes a list of structures and stores them in a HDF5 object.

    Parameters
    ----------
    structure_file_paths: List[pathlib.Path]
        List of paths to pdb files to be processed into frames
    output_path: pathlib.Path
        Path where dataset will be written.
    frame_edge_length: float
        The length of the edges of the frame in Angstroms.
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
    codec: object
        Codec object with encoding instructions.
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
                    codec,
                ),
            )
            for proc_i in range(processes)
        ]
        metadata = DatasetMetadata(
            make_frame_dataset_ver=MAKE_FRAME_DATASET_VER,
            frame_dims=(
                voxels_per_side,
                voxels_per_side,
                voxels_per_side,
                codec.encoder_length,
            ),
            atom_encoder=list(codec.atomic_labels),
            encode_cb=encode_cb,
            atom_filter_fn=str(atom_filter_fn),
            residue_encoder=list(standard_amino_acids.values()),
            frame_edge_length=frame_edge_length,
        )
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


def _select_pdb_chain(pdb_path: pathlib.Path, chain: str):
    """
    Select a chain from a pdb file. The chain will remove the original pdb file.
    At the moment we only support the selection of one chain at the time, meaning
    if you wanted to selected chain A and B of a PDB, you should write it twice eg.
    "6FMLA, 6FMLB".

    Parameters
    ----------
    pdb_path: pathlib.Path
        Path to the pdb structure.
    chain: str
        Chain to be selected for the pdb
    Returns
    -------
    chain_pdb: ampal.Assembly
        Ampal object with the selected chain
    """
    pdb_structure = ampal.load_pdb(pdb_path)
    # Check if PDB structure is container and select assembly
    if isinstance(pdb_structure, ampal.AmpalContainer):
        warnings.warn(
            f"Selecting the first state from the NMR structure {pdb_structure.id}"
        )
        pdb_structure = pdb_structure[0]

    chain_pdb = pdb_structure[chain]
    warnings.warn(f"ATTENTION: You selected chain {chain}, for PDB code {pdb_path}. We will replace the original PDB file with the selected chain. Remove the 5th letter of your PDB code if this is unwanted behaviour.")
    # Save chain to file:
    output_pdb_path = pathlib.Path(str(pdb_path.with_suffix("")) + chain + pdb_path.suffix)
    with open(output_pdb_path, "w") as f:
        f.write(chain_pdb.pdb)
    # Delete original file
    if pdb_path.exists():
        pdb_path.unlink()
    return chain_pdb


def _fetch_pdb(
    pdb_code: str,
    output_folder: pathlib.Path = PDB_PATH,
    pdb_request_url: str = PDB_REQUEST_URL,
    download_assembly: bool = True,
) -> pathlib.Path:
    """
    Downloads a specific pdb file into a specific folder.

    Parameters
    ----------
    pdb_code : str
        Code of the PDB file to be downloaded.
    output_folder : Path
        Output path to save the PDB file.
    pdb_request_url : str
        Base URL to download the PDB files.
    download_assembly: bool
        Whether to download the biological assembly file of the pdb.

    Returns
    -------
    output_path: Path
        Path to downloaded pdb

    """

    # Remove empty spaces
    pdb_code = pdb_code.strip(" ")

    assert (len(pdb_code) == 4) or (len(pdb_code) == 5), (
        f"Expected pdb code to be of length 4 or 5 (pdb+chain) but "
        f"got {len(pdb_code)}"
    )

    # Retrieve pdb:
    if download_assembly:
        pdb_code_with_extension = f"{pdb_code[:4]}.pdb1"
    else:
        pdb_code_with_extension = f"{pdb_code[:4]}.pdb"
    output_path = output_folder / pdb_code_with_extension
    urllib.request.urlretrieve(
        pdb_request_url + pdb_code_with_extension,
        filename=output_path,
    )
    # If PDB code is 5, user likely specified a chain
    if len(pdb_code) == 5:
        # Extract chain from string:
        chain = pdb_code[-1]
        _ = _select_pdb_chain(output_path, chain)

    return output_path


def download_pdb_from_csv_file(
    pdb_csv_file: pathlib.Path, pdb_outpath: pathlib.Path = PDB_PATH
):
    """
    Dowloads PDB functional unit files of structures from a csv file.

    Parameters
    ----------
    pdb_csv_file: pathlib.Path
        Path to the csv file with PDB codes.

    pdb_outpath: pathlib.Path
        Path output where PDBs will be saved to.

    Returns
    -------
    structure_file_paths: t.List[StrOrPath]
        List of strings / paths to the newly downloaded PDBs structures

    """
    with open(pdb_csv_file) as csv_file:
        protein_csv = csv.reader(csv_file, delimiter=",")
        pdb_list = next(protein_csv)
    # Check if pdb folder exists
    if pathlib.Path(pdb_outpath).exists():
        warnings.warn(
            f"{pdb_outpath} folder already exists. PDB files will be added next to already existing ones."
        )
    else:
        pathlib.Path(pdb_outpath).mkdir(parents=True, exist_ok=True)

    structure_file_paths = [
        _fetch_pdb(pdb_code, download_assembly=True, output_folder=pdb_outpath)
        for pdb_code in pdb_list
    ]

    return structure_file_paths


def make_frame_dataset(
    structure_files: t.List[StrOrPath],
    output_folder: StrOrPath,
    name: str,
    frame_edge_length: float,
    voxels_per_side: int,
    codec: object,
    atom_filter_fn: t.Callable[[ampal.Atom], bool] = default_atom_filter,
    pieces_filter_file: t.Optional[StrOrPath] = None,
    processes: int = 1,
    gzipped: bool = False,
    verbosity: int = 1,
    require_confirmation: bool = True,
    encode_cb: bool = True,
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
    codec: object
        Codec object with encoding instructions.
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
            if p.name.split(".")[0].upper().strip("PDB") in chain_filter_dict
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
        codec=codec,
    )
    return output_file_path


# }}}
