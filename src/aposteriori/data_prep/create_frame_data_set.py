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
from itertools import repeat
from multiprocessing import Pool

import ampal
import ampal.geometry as geometry
import h5py
import numpy as np
from ampal.amino_acids import residue_charge, polarity_Zimmerman, standard_amino_acids

from aposteriori.config import (
    ATOM_VANDERWAAL_RADII,
    MAKE_FRAME_DATASET_VER,
    PDB_REQUEST_URL,
    UNCOMMON_RESIDUE_DICT,
)


# {{{ Types
@dataclass
class ResidueResult:
    residue_id: str
    label: str
    encoded_residue: np.ndarray
    data: np.ndarray
    voxels_as_gaussian: bool
    rotamers: str


@dataclass
class DatasetMetadata:
    make_frame_dataset_ver: str
    frame_dims: t.Tuple[int, int, int, int]
    atom_encoder: t.List[str]
    encode_cb: bool
    atom_filter_fn: str
    residue_encoder: t.List[str]
    frame_edge_length: float
    voxels_as_gaussian: bool

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

    @classmethod
    def CNOCBCAQ(cls):
        return cls(["C", "N", "O", "CB", "CA", "Q"])

    @classmethod
    def CNOCBCAP(cls):
        return cls(["C", "N", "O", "CB", "CA", "P"])

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
        if atom_label in self.label_to_encoding.keys():
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

    def encode_gaussian_atom(
        self, atom_label: str, modifiers_triple: t.Tuple[float, float, float]
    ) -> (np.ndarray, int):
        """
        Encodes atom as a 3x3 gaussian with length of encoder length. Only the
        C, N and O atoms are represented in gaussian form. If Ca and Cb are
        encoded separately, they will be represented as a gaussian in their
        separate channel.

        Parameters
        ----------
        atom_label: str
            Label of the atom to be encoded.
        modifiers_triple: t.Tuple[float, float, float]
            Triple of the difference between the discretized coordinate and the
            undiscretized coordinate.

        Returns
        -------
        atom_encoding: np.ndarray
            Boolean array with atom encoding of shape (3, 3, 3, encoder_length,)

        """
        # Creating empty atom encoding:
        encoded_atom = np.zeros((3, 3, 3, self.encoder_length))
        # Attempt encoding:
        if atom_label.upper() in self.label_to_encoding.keys():
            atom_idx = self.label_to_encoding[atom_label.upper()]
        # Fix labels:
        elif self.encoder_length == 3:
            # In this scenario, the encoder is C,N,O so Cb and Ca are just C atoms
            if (atom_label.upper() == "CA") or (atom_label.upper() == "CB"):
                atom_label = "C"
                atom_idx = self.label_to_encoding[atom_label]
        elif self.encoder_length == 4:
            # In this scenario, Ca is a carbon atom but Cb has a separate channel
            if atom_label.upper() == "CA":
                atom_label = "C"
                atom_idx = self.label_to_encoding[atom_label]
        else:
            raise ValueError(
                f"{atom_label} not found in {self.atomic_labels} encoding."
            )
        # If label to encode is C, N, O:
        if atom_idx in ATOM_VANDERWAAL_RADII.keys():
            # Get encoding:
            atomic_radius = ATOM_VANDERWAAL_RADII[atom_idx]
            atom_to_encode = convert_atom_to_gaussian_density(
                modifiers_triple, atomic_radius
            )
            # Add to original atom:
            encoded_atom[:, :, :, atom_idx] += atom_to_encode
        # If label encodes Cb and Ca as separate channels (ie, not CNO):
        elif atom_label.upper() in self.label_to_encoding.keys():
            # Get encoding:
            atomic_radius = ATOM_VANDERWAAL_RADII[0]
            atom_to_encode = convert_atom_to_gaussian_density(
                modifiers_triple, atomic_radius
            )
            encoded_atom[:, :, :, atom_idx] += atom_to_encode
        else:
            raise ValueError(
                f"{atom_label} not found in {self.atomic_labels} encoding. Returning empty array."
            )

        return encoded_atom, atom_idx

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


def encode_cb_prevox(residue: ampal.Residue):
    """
    Encodes a Cb atom to all of the AMPAL residues before the voxelisation begins. The Cb is added to an average position
    calculated by averaging the Cb coordinates of the aligned frames for the 1QYS protein.

    Parameters
    ----------
    residue: ampal.Residue
        Focus residues that requires the Cb atom.

    """
    align_to_residue_plane(residue)
    encode_cb_to_ampal_residue(residue)
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


def convert_atom_to_gaussian_density(
    modifiers_triple: t.Tuple[float, float, float],
    wanderwaal_radius: float,
    range_val: int = 2,
    resolution: int = 999,
    optimized: bool = True,
):
    """
    Converts an atom at a coordinate, with specific modifiers due to a discretization,
    into a 3x3x3 gaussian density using the formula indicated  by Zhang et al., (2019) ProdCoNN.

    https://onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2Fprot.25868&file=prot25868-sup-0001-AppendixS1.pdf

    Parameters
    ----------
    modifiers_triple: t.Tuple[float, float, float]
        Triple of the difference between the discretized coordinate and the undiscretized coordinate.
    wanderwaal_radius: float
        Wanderwaal radius of the current atom being discretized.
    range_val: int
        Range values for the gaussian. Default = 2
    resolution: int
        Number of points selected between the `range_val` to calculate the density. Default = 999
    optimized: bool
        Whether to use an optimized algorithm to produce the gaussian. Default = True
    Returns
    -------
    norm_gaussian_frame: np.ndarray
        3x3x3 Frame encoding for a gaussian atom
    """
    if optimized:
        # Unpack x, y, z:
        x, y, z = modifiers_triple
        # Obtain x, y, z ranges for the gaussian
        x_range = np.linspace(-range_val + x, range_val - x, resolution)
        y_range = np.linspace(-range_val + y, range_val - y, resolution)
        z_range = np.linspace(-range_val + z, range_val - z, resolution)
        # Calculate density for each axis at each point
        x_vals, y_vals, z_vals = (
            np.exp(-1 * ((x_range - x) / wanderwaal_radius) ** 2, dtype=np.float16),
            np.exp(-1 * ((y_range - y) / wanderwaal_radius) ** 2, dtype=np.float16),
            np.exp(-1 * ((z_range - z) / wanderwaal_radius) ** 2, dtype=np.float16),
        )

        x_densities = []
        y_densities = []
        z_densities = []
        r = resolution // 3
        # Integrate to get area under the gaussian curve:
        for i in range(0, resolution, r):
            x_densities.append(np.trapz(x_vals[i : i + r], x_range[i : i + r]))
            y_densities.append(np.trapz(y_vals[i : i + r], y_range[i : i + r]))
            z_densities.append(np.trapz(z_vals[i : i + r], z_range[i : i + r]))
        # # Create grids for x, y and z :
        xyz_grids = np.meshgrid(x_densities, y_densities, z_densities)
        # The multiplication here is necessary so that e**x * e**y * e**z are equivalent to
        # e**(x + y + z)
        gaussian_frame = xyz_grids[0] * xyz_grids[1] * xyz_grids[2]

    else:
        gaussian_frame = np.zeros((3, 3, 3), dtype=float)
        xyz_coordinates = np.where(gaussian_frame == 0)
        # Identify the real (undiscretized) coordinates of the atom in the 3x3x3 matrix
        x, y, z = modifiers_triple
        x, y, z = x + 1, y + 1, z + 1
        # The transpose changes arrays of [y], [x], [z] into [y, x, z]
        for voxel_coord in np.array(xyz_coordinates).T:
            # Extract voxel coords:
            vy, vx, vz = voxel_coord
            # Calculate Density:
            voxel_density = np.exp(
                -((vx - x) ** 2 + (vy - y) ** 2 + (vz - z) ** 2)
                / wanderwaal_radius**2
            )
            # Add density to frame:
            gaussian_frame[vy, vx, vz] = voxel_density

    # Normalize so that values add up to 1:
    norm_gaussian_frame = gaussian_frame / np.sum(gaussian_frame)

    return norm_gaussian_frame


def calculate_atom_coord_modifier_within_voxel(
    atom: ampal.Atom,
    voxel_edge_length: float,
    indices: t.Tuple[int, int, int],
    adjust_by: int = 0,
) -> t.Tuple[float, float, float]:
    """
    Calculates modifiers lost during the discretization of atoms within the voxel.

    Assuming an atom coordinate was x = 2.6, after the discretization it will occupy
    the voxel center_voxel_x + int(2.6 / 0.57). So assuming center voxel of 10,
    the atom will occupy x = 10 + 5, though (2.6 / 0.57) is about 4.56.

    This means that our discretization is losing about 5 - 4.56 = 0.44 position.
    The gaussian representation aims at accounting for this approximation.

    Parameters
    ----------
    atom: ampal.Atom
        Atom x, y, z coordinates will be discretized based on `voxel_edge_length`.
    voxel_edge_length: float
        Edge length of the voxels that are mapped onto cartesian space.
    indices: t.Tuple[int, int, int],
        Triple containing discretized (x, y, z) coordinates of the atom.

    Returns
    -------
    modifiers_triple: t.Tuple[float, float, float]
        Triple containing modifiers for the (x, y, z) coordinates
    """
    # Extract discretized coordinates:
    dx, dy, dz = indices
    # Calculate modifiers:
    modifiers_triple = (
        dx - (atom.x / voxel_edge_length + adjust_by),
        dy - (atom.y / voxel_edge_length + adjust_by),
        dz - (atom.z / voxel_edge_length + adjust_by),
    )

    return modifiers_triple


def add_gaussian_at_position(
    main_matrix: np.ndarray,
    secondary_matrix: np.ndarray,
    atom_coord: t.Tuple[int, int, int],
    atom_idx: int,
    atomic_center: t.Tuple[int, int, int] = (1, 1, 1),
    normalize: bool = True,
) -> np.ndarray:
    """
    Adds a 3D array (of a gaussian atom) to a specific coordinate of a frame.

    Parameters
    ----------
    main_matrix: np.ndarray
        Frame 4D array VxVxVxE where V is the n. of voxels and E is the length of
        the encoder.
    secondary_matrix: np.ndarray
        3D matrix containing gaussian atom. Usually 3x3x3
    atom_coord: t.Tuple[int, int, int]
        Coordinates of the atom in voxel numbers
    atom_idx: int
        Denotes the atom position in the encoder (eg. C = 0)
    atomic_center: t.Tuple[int, int, int]
        Center of the atom within the Gaussian representation

    Returns
    -------
    density_frame: np.ndarray
        Frame 4D array with gaussian atom added into it.
    """

    # Copy the main matrix:
    density_frame = main_matrix

    # Remember in a 4D matrix, our 4th dimension is the length of the atom_encoder.
    # This means that to obtain a 3D frame we can just select array[:,:,:, ATOM_NUMBER]
    # Select 3D frame of the atom to be added:
    empty_frame_voxels = np.zeros(
        (density_frame[:, :, :, atom_idx].shape), dtype=np.float16
    )
    # Slice the atom density matrix:
    # This is necessary in case we are at the edge of the frame in which case we
    # need to cut the bits that will not be added.
    density_matrix_slice = secondary_matrix[
        max(atomic_center[0] - atom_coord[0], 0) : atomic_center[0]  # min y
        - atom_coord[0]
        + empty_frame_voxels.shape[0],  # max y
        max(atomic_center[1] - atom_coord[1], 0) : atomic_center[1]  # min x
        - atom_coord[1]
        + empty_frame_voxels.shape[1],  # max x
        max(atomic_center[2] - atom_coord[2], 0) : atomic_center[2]  # min z
        - atom_coord[2]
        + empty_frame_voxels.shape[2],  # max z
    ]
    # Normalize local densities by sum of all densities (so that they all sum up to 1):
    if normalize:
        density_matrix_slice /= np.sum(density_matrix_slice)
    # Slice the Frame to select the portion that contains the atom of interest:
    frame_slice = empty_frame_voxels[
        max(atom_coord[0] - int(density_matrix_slice.shape[0] / 2), 0) : max(
            atom_coord[0] - int(density_matrix_slice.shape[0] / 2), 0
        )
        + density_matrix_slice.shape[0],  # max y
        max(atom_coord[1] - int(density_matrix_slice.shape[1] / 2), 0) : max(
            atom_coord[1] - int(density_matrix_slice.shape[1] / 2), 0
        )
        + density_matrix_slice.shape[1],  # max x
        max(atom_coord[2] - int(density_matrix_slice.shape[2] / 2), 0) : max(
            atom_coord[2] - int(density_matrix_slice.shape[2] / 2), 0
        )
        + density_matrix_slice.shape[2],  # max z
    ]
    # Add atom density to the frame:
    frame_slice += density_matrix_slice
    # Add to original array:
    density_frame[:, :, :, atom_idx] += empty_frame_voxels

    return density_frame


def create_residue_frame(
    residue: ampal.Residue,
    frame_edge_length: float,
    voxels_per_side: int,
    encode_cb: bool,
    codec: object,
    voxels_as_gaussian: bool = False,
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
    voxels_as_gaussian: bool
        Whether to encode voxels as gaussians.

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
    if "P" in codec.atomic_labels:
        if residue.mol_letter in standard_amino_acids.keys():
            res_property = -1 if polarity_Zimmerman[residue.mol_letter] < 20 else 1
        else:
            res_property = 0
        # res_property = -1 if res_property < 20 else 1
    elif "Q" in codec.atomic_labels:
        res_property = residue_charge[residue.mol_letter]

    align_to_residue_plane(residue)

    frame = np.zeros(
        (voxels_per_side, voxels_per_side, voxels_per_side, codec.encoder_length),
    )
    # Change frame type to float if gaussian else use bool:
    frame = frame.astype(np.float16) if voxels_as_gaussian else frame.astype(np.bool)
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
        if not voxels_as_gaussian:
            assert frame[indices][0] == False, (
                f"Voxel should not be occupied: Currently "
                f"{frame[indices]}, "
                f"{ass.id}:{cha.id}:{res.id}:{atom.res_label}"
            )
            # If the voxel is a gaussian, there may be remnants of a nearby atom
            # hence this test would fail
        if not voxels_as_gaussian:
            if not atom.res_label == "CB":
                np.testing.assert_array_equal(
                    frame[indices], np.array([False] * len(frame[indices]), dtype=bool)
                )
        # Encode atoms:
        if voxels_as_gaussian:
            modifiers_triple = calculate_atom_coord_modifier_within_voxel(
                atom, voxel_edge_length, indices, adjust_by=voxels_per_side // 2
            )
            # Get Gaussian encoding
            gaussian_matrix, atom_idx = Codec.encode_gaussian_atom(
                codec, atom.res_label, modifiers_triple
            )
            gaussian_atom = gaussian_matrix[:, :, :, atom_idx]
            # Add at position:
            frame = add_gaussian_at_position(
                main_matrix=frame,
                secondary_matrix=gaussian_atom,
                atom_coord=indices,
                atom_idx=atom_idx,
            )
            if (
                "Q" in codec.atomic_labels
                or "P" in codec.atomic_labels
                and res_property != 0
            ):
                gaussian_atom = gaussian_matrix[:, :, :, atom_idx] * float(res_property)
                # Add at position:
                frame = add_gaussian_at_position(
                    main_matrix=frame,
                    secondary_matrix=gaussian_atom,
                    atom_coord=indices,
                    atom_idx=5,
                    normalize=False,
                )
        else:
            # Encode atom as voxel:
            frame[indices] = Codec.encode_atom(codec, atom.res_label)
            if (
                "Q" in codec.atomic_labels
                or "P" in codec.atomic_labels
                and res_property != 0
            ):
                frame[indices] = res_property
    centre = voxels_per_side // 2
    # Check whether central atom is C:
    if "CA" in codec.atomic_labels:
        if voxels_as_gaussian:
            np.testing.assert_array_less(frame[centre, centre, centre][4], 1)
            assert (
                0 < frame[centre, centre, centre][4] <= 1
            ), f"The central atom value should be between 0 and 1 but was {frame[centre, centre, centre][4]}"
        else:
            assert (
                frame[centre, centre, centre][4] == 1
            ), f"The central atom should be Carbon, but it is {frame[centre, centre, centre]}."
    else:
        if voxels_as_gaussian:
            np.testing.assert_array_less(frame[centre, centre, centre][0], 1)
            assert (
                0 < frame[centre, centre, centre][0] <= 1
            ), f"The central atom value should be between 0 and 1 but was {frame[centre, centre, centre][0]}"

        else:
            assert (
                frame[centre, centre, centre][0] == 1
            ), f"The central atom should be Carbon, but it is {frame[centre, centre, centre]}."
    return frame


def voxelise_assembly(
    assembly,
    atom_filter_fn,
    name,
    chain_filter_list,
    verbosity,
    chain_dict,
    frame_edge_length,
    voxels_per_side,
    encode_cb,
    codec,
    voxels_as_gaussian,
    tag_rotamers,
):
    if tag_rotamers:
        if isinstance(assembly, ampal.AmpalContainer):
            # For each assembly:
            for real_assembly in assembly:
                # For each monomer in the assembly:
                for monomer in real_assembly:
                    if isinstance(monomer, ampal.Polypeptide):
                        monomer.tag_sidechain_dihedrals()
        if isinstance(assembly, ampal.Assembly):
            # For each monomer in the assembly:
            for monomer in assembly:
                if isinstance(monomer, ampal.Polypeptide):
                    monomer.tag_sidechain_dihedrals()
        elif isinstance(assembly, ampal.Polypeptide):
            assembly.tag_sidechain_dihedrals()

    # Filters atoms not related to assembly:
    total_atoms = len(list(assembly.get_atoms()))
    for atom in assembly.get_atoms():
        if not atom_filter_fn(atom):
            del atom.parent.atoms[atom.res_label]
            del atom
    if "CB" in codec.atomic_labels:
        if encode_cb:
            for chain in assembly:
                if not isinstance(chain, ampal.Polypeptide):
                    continue
                for residue in chain:
                    encode_cb_prevox(residue)
    remaining_atoms = len(list(assembly.get_atoms()))
    print(f"{name}: Filtered {total_atoms - remaining_atoms} of {total_atoms} atoms.")
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
        # Loop through each residue, voxels:
        for residue in chain:
            if isinstance(residue, ampal.Residue):
                # Create voxelised frame:
                array = create_residue_frame(
                    residue=residue,
                    frame_edge_length=frame_edge_length,
                    voxels_per_side=voxels_per_side,
                    encode_cb=encode_cb,
                    codec=codec,
                    voxels_as_gaussian=voxels_as_gaussian,
                )
                encoded_residue = encode_residue(residue.mol_code)
                if "rotamers" in list(residue.tags):
                    if any(v is None for v in residue.tags["rotamers"]):
                        rota = "NAN"
                    else:
                        rota = "".join(
                            np.array(residue.tags["rotamers"], dtype=str).tolist()
                        )
                else:
                    rota = "NAN"
                # Save results:
                chain_dict[chain.id].append(
                    ResidueResult(
                        residue_id=str(residue.id),
                        label=residue.mol_code,
                        encoded_residue=encoded_residue,
                        data=array,
                        voxels_as_gaussian=voxels_as_gaussian,
                        rotamers=rota,
                    )
                )
                if verbosity > 1:
                    print(f"{name}:\t\tAdded residue {chain.id}:{residue.id}.")
        if verbosity > 0:
            print(f"{name}:\tFinished processing chain {chain.id}.")

    return (name, chain_dict)


def create_frames_from_structure(
    structure_path: pathlib.Path,
    frame_edge_length: float,
    voxels_per_side: int,
    atom_filter_fn: t.Callable[[ampal.Atom], bool],
    chain_filter_list: t.Optional[t.List[str]],
    is_pdb_gzipped: bool,
    verbosity: int,
    encode_cb: bool,
    codec: object,
    voxels_as_gaussian: bool,
    voxelise_all_states: bool,
    tag_rotamers: bool,
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
    is_pdb_gzipped: bool
        Indicates if structure files are gzipped or not.
    verbosity: int
        Level of logging sent to std out.
    encode_cb: bool
        Whether to encode the Cb at an average position in the frame. If
        True, it will not be filtered by the `atom_filter_fn`.
    codec: object
        Codec object with encoding instructions.
    voxels_as_gaussian: bool
        Whether to encode voxels as gaussians.
    voxelise_all_states: bool
        Whether to voxelise only the first state of the NMR structure (False) or all of them (True).
    """
    name = structure_path.name.split(".")[0]
    chain_dict: ChainDict = {}
    if is_pdb_gzipped:
        with gzip.open(str(structure_path), "rb") as inf:
            assembly = ampal.load_pdb(inf.read().decode(), path=False)
    else:
        assembly = ampal.load_pdb(str(structure_path))
    # Deals with structures from NMR as ampal returns Container of Assemblies
    if isinstance(assembly, ampal.AmpalContainer) and voxelise_all_states:
        if verbosity > 1:
            warnings.warn(f"Voxelising all states from the NMR structure {assembly.id}")
        result = []
        for i, curr_assembly in enumerate(assembly):
            curr_result = voxelise_assembly(
                curr_assembly,
                atom_filter_fn,
                name + f"_{i}",
                chain_filter_list,
                verbosity,
                chain_dict,
                frame_edge_length,
                voxels_per_side,
                encode_cb,
                codec,
                voxels_as_gaussian,
                tag_rotamers,
            )

            result.append(curr_result)
    else:
        if isinstance(assembly, ampal.AmpalContainer):
            if verbosity > 1:
                warnings.warn(
                    f"Selecting the first state from the NMR structure {assembly.id}"
                )
            assembly = assembly[0]
        result = voxelise_assembly(
            assembly,
            atom_filter_fn,
            name,
            chain_filter_list,
            verbosity,
            chain_dict,
            frame_edge_length,
            voxels_per_side,
            encode_cb,
            codec,
            voxels_as_gaussian,
            tag_rotamers,
        )

    return result


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
    is_pdb_gzipped: bool,
    verbosity: int,
    encode_cb: bool,
    codec: object,
    voxels_as_gaussian: bool,
    voxelise_all_states: bool,
    tag_rotamers: bool,
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
                is_pdb_gzipped,
                verbosity,
                encode_cb,
                codec,
                voxels_as_gaussian=voxels_as_gaussian,
                voxelise_all_states=voxelise_all_states,
                tag_rotamers=tag_rotamers,
            )
        except Exception as e:
            result = str(e)
        if isinstance(result, str):
            errors[str(structure_path)] = result
        elif isinstance(result, list):
            for curr_res in result:
                result_queue.put(curr_res)
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
    gzip_compression: bool,
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
                    # This is required as at times the pdb does not have a chain name:
                    chain_id = "A" if not chain_id else chain_id
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
                        # Change type of voxel saved to hdf5 file depending on type of voxel used:
                        voxel_output_type = (
                            float if metadata_dict["voxels_as_gaussian"] else bool
                        )
                        res_dataset = chain_group.create_dataset(
                            res_result.residue_id,
                            data=res_result.data,
                            dtype=voxel_output_type,
                            compression="gzip" if gzip_compression else None,
                        )
                        res_dataset.attrs["label"] = res_result.label
                        res_dataset.attrs["rotamers"] = res_result.rotamers
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
    is_pdb_gzipped: bool,
    verbosity: int,
    encode_cb: bool,
    codec: object,
    voxels_as_gaussian: bool,
    gzip_compression: bool = True,
    voxelise_all_states: bool = True,
    tag_rotamers: bool = False,
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
    is_pdb_gzipped: bool
        Indicates if structure files are gzipped or not.
    verbosity: int
        Level of logging sent to std out.
    encode_cb: bool
        Whether to encode the Cb at an average position in the frame.
    codec: object
        Codec object with encoding instructions.
    voxels_as_gaussian: bool
        Whether to encode voxels as gaussians.
    voxelise_all_states: bool
        Whether to voxelise only the first state of the NMR structure (False) or all of them (True).
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
                    is_pdb_gzipped,
                    verbosity,
                    encode_cb,
                    codec,
                    voxels_as_gaussian,
                    voxelise_all_states,
                    tag_rotamers,
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
            voxels_as_gaussian=voxels_as_gaussian,
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
                gzip_compression,
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


def _select_pdb_chain(
    output_pdb_path: pathlib.Path,
    pdb_structure: ampal.Assembly,
    pdb_name: str,
    chain: str,
    verbosity: int,
    return_chain_path: bool = True,
    nmr_state: int = None,
) -> (pathlib.Path, ampal.Assembly):
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
    return_chain_path: bool
        Whether to return new_chain_path (True), or return chain (False)
    Returns
    -------
    chain_pdb: ampal.Assembly
        Ampal object with the selected chain
    output_pdb_path: pathlib.Path
        Output path with chain
    """
    # Check if PDB structure is container and select assembly:
    chain_pdb = pdb_structure[chain]
    if verbosity > 1:
        warnings.warn(
            f"ATTENTION: You selected chain {chain}, for PDB code {pdb_name}. We will replace the original PDB file with the selected chain. Remove the 5th letter of your PDB code if this is unwanted behaviour."
        )
    # Save chain to file:
    pdb_name += chain
    if nmr_state:
        pdb_name += f"_{nmr_state}"
    # Extract pdb extension:
    ext_pdb = output_pdb_path.suffix
    chain_pdb_path = output_pdb_path.parent / f"{pdb_name}{ext_pdb}"
    # Save chain file
    with open(chain_pdb_path, "w") as f:
        f.write(chain_pdb.pdb)
    # Delete original file
    if output_pdb_path.exists():
        output_pdb_path.unlink()

    if return_chain_path:
        return chain_pdb_path
    else:
        return chain_pdb


def _fetch_pdb(
    pdb_code: str,
    verbosity: int,
    output_folder: pathlib.Path,
    download_assembly: bool = True,
    voxelise_all_states: bool = False,
    pdb_request_url: str = PDB_REQUEST_URL,
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
    if voxelise_all_states:
        download_assembly = False
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
    # Load structure
    pdb_structure = ampal.load_pdb(output_path)
    # If PDB code is 5, user likely specified a chain
    if len(pdb_code) == 5:
        # Extract chain from string:
        chain = pdb_code[-1]
        # Check if PDB structure is container and select assembly:
        if isinstance(pdb_structure, ampal.AmpalContainer) and len(pdb_structure) > 1:
            if voxelise_all_states:
                out_paths = []
                for i, curr_structure in enumerate(pdb_structure):
                    curr_outpath = _select_pdb_chain(
                        output_pdb_path=output_path,
                        pdb_structure=curr_structure,
                        pdb_name=pdb_code[:4],
                        chain=chain,
                        verbosity=verbosity,
                        nmr_state=i,
                    )
                    out_paths.append(curr_outpath)
                output_path = out_paths
            else:
                if verbosity > 1:
                    warnings.warn(
                        f"Selecting the first state from the NMR structure {pdb_structure.id}"
                    )
                pdb_structure = pdb_structure[0]
                output_path = _select_pdb_chain(
                    output_pdb_path=output_path,
                    pdb_structure=pdb_structure,
                    pdb_name=pdb_code[:4],
                    chain=chain,
                    verbosity=verbosity,
                )
        else:
            output_path = _select_pdb_chain(
                output_pdb_path=output_path,
                pdb_structure=pdb_structure,
                pdb_name=pdb_code[:4],
                chain=chain,
                verbosity=verbosity,
            )
    elif len(pdb_code) == 4:
        if voxelise_all_states:
            if (
                isinstance(pdb_structure, ampal.AmpalContainer)
                and len(pdb_structure) > 1
            ):
                ext_pdb = output_path.suffix
                out_paths = []

                for i, curr_structure in enumerate(pdb_structure):
                    state_pdb_path = output_path.parent / f"{pdb_code[:4]}_{i}{ext_pdb}"
                    # Save structure file:
                    with open(state_pdb_path, "w") as f:
                        f.write(curr_structure.pdb)
                    out_paths.append(state_pdb_path)
                output_path = out_paths

    return output_path


def filter_structures_by_blacklist(
    structure_files: t.List[pathlib.Path], blacklist_csv_file: pathlib.Path
) -> t.List[StrOrPath]:
    """
    Filters structures contained in a blacklist csv file.

    Parameters
    ----------
    structure_files: List[pathlib.Path]
        List of paths to pdb files to be processed into frames
    blacklist_csv_file: pathlib.Path
        Path to blacklist csv.

    Returns
    -------
    filtered_structure_files: List[pathlib.Path]
        List of filtered paths to pdb files to be processed into frames
    """

    # Open blacklist file and extract list:
    with open(blacklist_csv_file) as csv_file:
        blacklist_csv = csv.reader(csv_file, delimiter=",")
        blacklist = set()
        # Reading to set to make sure entries are unique:
        for b_pdb in next(blacklist_csv):
            curr_pdb = b_pdb.strip(" ").lower()
            assert (
                len(curr_pdb) == 4 or len(curr_pdb) == 5
            ), f"Expected PDB to be length of 4 or 5 but found {len(curr_pdb)}"
            blacklist.add(curr_pdb[:4])

    filtered_structure_files = []
    # Loop through structures
    for structure in structure_files:
        # Remove extension: (deals with double extension too)
        curr_pdb = structure.stem.split(".")[0].lower()
        assert (
            len(curr_pdb) == 4 or len(curr_pdb) == 5
        ), f"Expected PDB to be length of 4 or 5 but found {len(curr_pdb)}"
        # if pdb not in blacklist
        if curr_pdb[:4] not in blacklist:
            # keep it:
            filtered_structure_files.append(structure)
    # Calculate difference
    old_length = len(structure_files)
    new_length = len(filtered_structure_files)
    print(
        f"Filtered {old_length - new_length} structures from the original {old_length} structures"
    )

    return filtered_structure_files


def download_pdb_from_csv_file(
    pdb_csv_file: pathlib.Path,
    verbosity: int,
    pdb_outpath: pathlib.Path,
    workers: int,
    voxelise_all_states: bool,
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

    # Use multiprocessing to download .pdb files faster
    with Pool(processes=workers) as p:
        structure_file_paths = p.starmap(
            _fetch_pdb,
            zip(
                pdb_list,
                repeat(verbosity),
                repeat(pdb_outpath),
                repeat(True),
                repeat(voxelise_all_states),
            ),
        )
        p.close()

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
    is_pdb_gzipped: bool = False,
    verbosity: int = 1,
    require_confirmation: bool = True,
    encode_cb: bool = True,
    voxels_as_gaussian: bool = False,
    blacklist_csv: pathlib.Path = None,
    gzip_compression: bool = True,
    voxelise_all_states: bool = True,
    tag_rotamers: bool = False,
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
    is_pdb_gzipped: bool
        Indicates if structure files are gzipped or not.
    verbosity: int
        Level of logging sent to std out.
    require_confirmation: bool
        If True, the user will be prompted to start creating the dataset.
    encode_cb: bool
        Whether to encode the Cb at an average position in the frame.
    voxels_as_gaussian: bool
        Whether the voxels are encoded as a floating point of a gaussian (True) or boolean (False).
        This converts an atom at a coordinate, with specific modifiers due to discretization,
        into a 3x3x3 gaussian density using the formula indicated by Zhang et al., (2019) ProdCoNN.

        https://onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2Fprot.25868&file=prot25868-sup-0001-AppendixS1.pdf
    blacklist_csv: StrOrPath
        Path to blacklist csv file.

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
    # Filter by blacklist:
    if blacklist_csv:
        # If blacklist path exists:
        if pathlib.Path(blacklist_csv).exists():
            filtered_structure_files = filter_structures_by_blacklist(
                structure_file_paths, pathlib.Path(blacklist_csv)
            )
        else:
            # Blacklist not fount:
            raise ValueError(f"Blacklist Path {blacklist_csv} not found.")
    else:
        filtered_structure_files = structure_file_paths

    output_file_path = pathlib.Path(output_folder) / (name + ".hdf5")
    total_files = len(filtered_structure_files)
    processed_files = 0
    number_of_frames = 0

    print(f"Will attempt to process {total_files} structure file/s.")
    print(f"Output file will be written to `{output_file_path.resolve()}`.")
    voxel_edge_length = frame_edge_length / voxels_per_side
    max_voxel_distance = np.sqrt(voxel_edge_length**2 * 3)
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
        structure_file_paths=filtered_structure_files,
        output_path=output_file_path,
        frame_edge_length=frame_edge_length,
        voxels_per_side=voxels_per_side,
        processes=processes,
        atom_filter_fn=atom_filter_fn,
        chain_filter_dict=chain_filter_dict,
        is_pdb_gzipped=is_pdb_gzipped,
        verbosity=verbosity,
        encode_cb=encode_cb,
        codec=codec,
        voxels_as_gaussian=voxels_as_gaussian,
        gzip_compression=gzip_compression,
        voxelise_all_states=voxelise_all_states,
        tag_rotamers=tag_rotamers,
    )
    return output_file_path


# }}}
