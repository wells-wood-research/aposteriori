"""Tools for creating a frame data set.

In this type of data set, all individual entries are stored separately in a flat
structure.
"""

import glob
import pathlib
import typing as t

import ampal
import ampal.geometry as geometry
import click
import h5py
import numpy as np


StrOrPath = t.Union[str, pathlib.Path]


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


def within_frame(frame_edge_length: float, atom: ampal.Atom) -> bool:
    """Tests if an atom is within the `radius` of the origin."""
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


def create_residue_frame(
    residue: ampal.Residue, frame_edge_length: float, voxels_per_side: int,
) -> np.ndarray:
    """Creates a discreet representation of a volume of space around a residue.
    
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
    
    Returns
    -------
    frame: ndarray
        Numpy array containing the discreet representation of a cube of space around the
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
    # create an empty array for discreet frame
    frame = np.zeros(
        (voxels_per_side, voxels_per_side, voxels_per_side), dtype=np.uint8
    )
    frame.fill(0)
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
        assert frame[indices] == 0, (
            f"Voxel should not be occupied: Currently "
            f"{frame[indices]}, "
            f"{ass.id}:{cha.id}:{res.id}:{atom.res_label}"
        )
        element_data = ampal.data.ELEMENT_DATA[atom.element.capitalize()]
        frame[indices] = element_data["atomic number"]
    centre = voxels_per_side // 2
    assert (
        frame[centre, centre, centre] == 6
    ), f"The central atom should be carbon, but it is {frame[centre, centre, centre]}."
    return frame


def default_atom_filter(atom: ampal.Atom) -> bool:
    """Filters for all heavy protein backbone atoms."""
    backbone_atoms = ("N", "CA", "C", "O")
    if atom.element == "H":
        return False
    elif isinstance(atom.parent, ampal.Residue) and (atom.res_label in backbone_atoms):
        return True
    else:
        return False


def make_dataset(
    structure_files: t.Iterable[StrOrPath],
    output_folder: StrOrPath,
    name: str,
    frame_edge_length: float,
    voxels_per_side: int,
    atom_filter_fn: t.Callable[[ampal.Atom], bool] = default_atom_filter,
    verbosity: int = 1,
) -> pathlib.Path:
    """Creates a data set of voxelized amino acid frames.

    Parameters
    ----------
    structure_files: List[str or pathlib.Path]
        List of paths to pdb files to be processed into frames
    output_folder: StrOrPath
        Path to folder where output will be written.
    name: str
        Name used for the data set file, `.hd5` will be appended.
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
    verbosity: int
        Level of logging sent to std out.

    Returns
    -------
    output_file_path: pathlib.Path
        A path to the location of the output dataset.
    """
    assert len(structure_files) > 0, "Aborting, no structure files defined."
    assert (
        voxels_per_side % 2
    ), "`voxels-per-side` must be odd, so that the CA is centred."
    structure_file_paths = [pathlib.Path(x) for x in structure_files]
    output_file_path = pathlib.Path(output_folder) / (name + ".hdf5")
    total_files = len(structure_file_paths)
    processed_files = 0
    number_of_frames = 0
    print(f"Output file will be written to `{output_file_path.resolve()}`.")
    with h5py.File(output_file_path, "w") as hd5:
        for structure_path in structure_file_paths:
            print(f"Processing `{structure_path}`...")
            assembly = ampal.load_pdb(str(structure_path))
            total_atoms = len(list(assembly.get_atoms()))
            for atom in assembly.get_atoms():
                if not atom_filter_fn(atom):
                    del atom.parent.atoms[atom.res_label]
                    del atom
            remaining_atoms = len(list(assembly.get_atoms()))
            print(f"Filtered {total_atoms-remaining_atoms} of {total_atoms} atoms.")
            pdb_group = hd5.create_group(structure_path.stem)
            for chain in assembly:
                if not isinstance(chain, ampal.Polypeptide):
                    if verbosity > 0:
                        print(f"\tIgnoring non-polypeptide chain ({chain.id}).")
                    continue
                if verbosity > 0:
                    print(f"\tProcessing chain {chain.id}...")
                chain_group = pdb_group.create_group(chain.id)
                for residue in chain:
                    if isinstance(residue, ampal.Residue):
                        array = create_residue_frame(
                            residue, frame_edge_length, voxels_per_side
                        )
                        dataset = chain_group.create_dataset(
                            str(residue.id),
                            (voxels_per_side, voxels_per_side, voxels_per_side),
                            dtype="u8",
                            data=array,
                        )
                        dataset.attrs["mol_code"] = residue.mol_code
                        number_of_frames += 1
                        if verbosity > 1:
                            print(f"\t\tAdded residue {chain.id}:{residue.id}.")
                if verbosity > 0:
                    print(f"\tFinished processing chain {chain.id}.")
            processed_files += 1
            print(f"Finished processing `{structure_path}`.")
            print(f"Files processed {processed_files}/{total_files}.")
    print(
        f"Created frame data set at `{output_file_path.resolve()}` containing "
        f"{number_of_frames} frames."
    )
    return output_file_path


@click.command()
@click.argument(
    "structure_files", type=click.Path(exists=True, readable=True), nargs=-1
)
@click.option(
    "-o",
    "--output-folder",
    type=click.Path(),
    default=".",
    help=("Path to folder where output will be written."),
)
@click.option(
    "-n",
    "--name",
    type=str,
    default="frame_data_set",
    help=(
        "Name used for the data set file, the `.hdf5` extension does not need to be "
        "included as it will be appended."
    ),
)
@click.option(
    "--frame-edge-length",
    type=float,
    default=12.0,
    help=(
        "Edge length of the cube of space around each residue that will be voxelized."
    ),
)
@click.option(
    "--voxels-per-side",
    type=int,
    default=21,
    help=(
        "The number of voxels per side of the frame. This will give a final cube of "
        "`voxels-per-side`^3."
    ),
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
def cli(
    structure_files: t.List[str],
    output_folder: str,
    name: str,
    frame_edge_length: float,
    voxels_per_side: int,
    verbose: int,
):
    """Creates a data set of voxelized amino acid frames.

    A frame refers to a region of space around an amino acid. Every
    residue in the input structure(s), a cube of space around the region
    (with an edge length equal to `--`, default 6 Å),
    will be mapped to discreet space, with a defined number of voxels per
    edge (equal to `--voxels-per-side`, default = 21).
    """
    output_file_path = make_dataset(
        structure_files,
        output_folder,
        name,
        frame_edge_length,
        voxels_per_side,
        default_atom_filter,
        verbose,
    )
    return


if __name__ == "__main__":
    # The cli will be run if this file is invoked directly
    # It is also hooked up as a script in `pyproject.toml`
    cli()

