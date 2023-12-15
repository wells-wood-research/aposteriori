import pathlib
import sys
import typing as t
import warnings

import click

from aposteriori.data_prep.create_frame_data_set import (
    Codec,
    StrOrPath,
    default_atom_filter,
    download_pdb_from_csv_file,
    make_frame_dataset,
)


# {{{ CLI
@click.command()
@click.argument(
    "structure_file_folder",
    type=str,
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
    "--is_pdb_gzipped",
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
        "Encode the Cb at an average position (-0.741287356, -0.53937931, -1.224287356) in the aligned frame, even for Glycine residues. Default = True"
    ),
)
@click.option(
    "-ae",
    "--atom_encoder",
    type=click.Choice(["CNO", "CNOCB", "CNOCBCA", "CNOCBCAQ", "CNOCBCAP"]),
    default="CNO",
    required=True,
    help=(
        "Encodes atoms in different channels, depending on atom types. Default is CNO, other options are ´CNOCB´ and `CNOCBCA` to encode the Cb or Cb and Ca in different channels respectively."
    ),
)
@click.option(
    "-d",
    "--download_file",
    type=click.Path(exists=True, readable=True),
    help=(
        "Path to csv file with PDB codes to be voxelised. The biological assembly will be used for download. PDB codes will be downloaded the /pdb/ folder."
    ),
)
@click.option(
    "-g",
    "--voxels_as_gaussian",
    type=bool,
    default=False,
    help=(
        "Boolean - whether to encode voxels as gaussians (True) or voxels (False). The gaussian representation uses the wanderwaal's radius of each atom using the formula e^(-x^2) where x is Vx - x)^2 + (Vy - y)^2) + (Vz - z)^2)/ r^2 and  (Vx, Vy, Vz) is the position of the voxel in space. (x, y, z) is the position of the atom in space, r is the Van der Waal’s radius of the atom. They are then normalized to add up to 1."
    ),
)
@click.option(
    "-b",
    "--blacklist_csv",
    type=click.Path(exists=True, readable=True),
    help=("Path to csv file with structures to be removed."),
)
@click.option(
    "-comp",
    "--compression_gzip",
    type=bool,
    default=True,
    help=("Whether to comrpess the dataset with gzip compression."),
)
@click.option(
    "-vas",
    "--voxelise_all_states",
    type=bool,
    default=False,
    help=(
        "Whether to voxelise only the first state of the NMR structure (False) or all of them (True)."
    ),
)
@click.option(
    "-rot",
    "--tag_rotamers",
    type=bool,
    default=False,
    help=("Whether to tag rotamer information to the frame (True) or not (False)."),
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
    is_pdb_gzipped: bool,
    recursive: bool,
    verbose: int,
    encode_cb: bool,
    atom_encoder: str,
    download_file: str,
    voxels_as_gaussian: bool,
    blacklist_csv: str,
    compression_gzip: bool,
    voxelise_all_states: bool,
    tag_rotamers: bool,
):
    """Creates a dataset of voxelized amino acid frames.

    A frame refers to a region of space around an amino acid. For every
    residue in the input structure(s), a cube of space around the region
    (with an edge length equal to `--frame_edge_length`, default 12 Å),
    will be mapped to discrete space, with a defined number of voxels per
    edge (equal to `--voxels-per-side`, default = 21).

    Basic Usage:

    `make-frame-dataset $path_to_folder_with_pdb/`

    eg. `make-frame-dataset tests/testing_files/pdb_files/`

    This command will make a tiny dataset in the current directory `test_dataset.hdf5`,
    containing all residues of the structures in the folder.

    Globs can be used to define the structure files to be processed.
    `make-frame-dataset pdb_files/**/*.pdb` would include all `.pdb` files in all
    subdirectories of the `pdb_files` directory.

    You can process gzipped pdb files, but the program assumes that the format
    of the file name is similar to `1mkk.pdb.gz`. If you have more complex
    requirements than this, we recommend using this library directly from
    Python rather than through this CLI.

    The hdf5 object itself is like a Python dict. The structure is
    simple:

    └─[pdb_code] Contains a number of subgroups, one for each chain.
      └─[chain_id] Contains a number of subgroups, one for each residue.
        └─[residue_id] voxels_per_side^3 array of ints, representing element number.
          └─.attrs['label'] Three-letter code for the residue.
          └─.attrs['encoded_residue'] One-hot encoding of the residue.
    └─.attrs['make_frame_dataset_ver']: str - Version used to produce the dataset.
    └─.attrs['frame_dims']: t.Tuple[int, int, int, int] - Dimentsions of the frame.
    └─.attrs['atom_encoder']: t.List[str] - Lables used for the encoding (eg, ["C", "N", "O"]).
    └─.attrs['encode_cb']: bool - Whether a Cb atom was added at the avg position of (-0.741287356, -0.53937931, -1.224287356).
    └─.attrs['atom_filter_fn']: str - Function used to filter the atoms in the frame.
    └─.attrs['residue_encoder']: t.List[str] - Ordered list of residues corresponding to the encoding used.
    └─.attrs['frame_edge_length']: float - Length of the frame in Angstroms (A)
    └─.attrs['voxels_as_gaussian']: bool - Whether the voxels are encoded as a floating point of a gaussian (True) or boolean (False)

    So hdf5['1ctf']['A']['58'] would be an array for the voxelized.
    """
    # If a download file is specified, open the file and download
    if download_file and pathlib.Path(download_file).exists():
        structure_files: t.List[StrOrPath] = download_pdb_from_csv_file(
            pdb_csv_file=pathlib.Path(download_file),
            pdb_outpath=pathlib.Path(output_folder),
            verbosity=verbose,
            workers=processes,
            voxelise_all_states=voxelise_all_states,
        )
        # TODO check if structure files is a flat list
    else:
        # Extract all the PDBs in folder:
        if pathlib.Path(structure_file_folder).exists():
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
        else:
            warnings.warn(
                f"{structure_file_folder} file not found. Did you specify the -d argument for the download file? If so, check your spelling."
            )
            sys.exit()
    # Create Codec:
    if atom_encoder == "CNO":
        codec = Codec.CNO()
    elif atom_encoder == "CNOCB":
        codec = Codec.CNOCB()
    elif atom_encoder == "CNOCBCA":
        codec = Codec.CNOCBCA()
    elif atom_encoder == "CNOCBCAQ":
        codec = Codec.CNOCBCAQ()
    elif atom_encoder == "CNOCBCAP":
        codec = Codec.CNOCBCAP()
    else:
        assert atom_encoder in [
            "CNO",
            "CNOCB",
            "CNOCBCA",
            "CNOCBCAQ",
            "CNOCBCAP",
        ], f"Expected encoder to be CNO, CNOCB, CNOCBCA, CNOCBCAQ, CNOCBCAP, but got {atom_encoder}"

    make_frame_dataset(
        structure_files=structure_files,
        output_folder=output_folder,
        name=name,
        frame_edge_length=frame_edge_length,
        voxels_per_side=voxels_per_side,
        codec=codec,
        atom_filter_fn=default_atom_filter,
        pieces_filter_file=pieces_filter_file,
        processes=processes,
        is_pdb_gzipped=is_pdb_gzipped,
        verbosity=verbose,
        encode_cb=encode_cb,
        voxels_as_gaussian=voxels_as_gaussian,
        blacklist_csv=blacklist_csv,
        gzip_compression=compression_gzip,
        voxelise_all_states=voxelise_all_states,
        tag_rotamers=tag_rotamers,
    )
    return


# }}}


if __name__ == "__main__":
    # The cli will be run if this file is invoked directly
    # It is also hooked up as a script in `pyproject.toml`
    cli()
