# aposteriori

DNN based protein design.

![CI](https://github.com/wells-wood-research/aposteriori/workflows/CI/badge.svg)

## Installation

### PyPI

_Coming soon..._

```sh
pip install aposteriori
```

## Creating a Data Set

There are two ways to create a dataset using `aposteriori`: through the Python API in
`aposteriori.make_frame_dataset` or using the command line tool `make-frame-dataset` that
installs along side the module:

```sh
make-frame-dataset tests/testing_files/pdb_files/
```

Check the `make-frame-dataset` help page for more details on its usage:

```sh
Usage: make-frame-dataset [OPTIONS] STRUCTURE_FILE_FOLDER

  Creates a dataset of voxelized amino acid frames.

  A frame refers to a region of space around an amino acid. For every
  residue in the input structure(s), a cube of space around the region (with
  an edge length equal to `--frame_edge_length`, default 12 Å), will be
  mapped to discrete space, with a defined number of voxels per edge (equal
  to `--voxels-per-side`, default = 21).

  Basic Usage:

  `make-frame-dataset $path_to_folder_with_pdb/`

  eg. `make-frame-dataset tests/testing_files/pdb_files/`

  This command will make a tiny dataset in the current directory
  `test_dataset.hdf5`, containing all residues of the structures in the
  folder.

  Globs can be used to define the structure files to be processed. `make-
  frame-dataset pdb_files/**/*.pdb` would include all `.pdb` files in all
  subdirectories of the `pdb_files` directory.

  You can process gzipped pdb files, but the program assumes that the format
  of the file name is similar to `1mkk.pdb.gz`. If you have more complex
  requirements than this, we recommend using this library directly from
  Python rather than through this CLI.

  The hdf5 object itself is like a Python dict. The structure is simple:

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

  So hdf5['1ctf']['A']['58'] would be an array for the voxelized.

Options:
  -o, --output-folder PATH        Path to folder where output will be written.
                                  Default = `.`

  -n, --name TEXT                 Name used for the dataset file, the `.hdf5`
                                  extension does not need to be included as it
                                  will be appended. Default = `frame_dataset`

  -e, --extension TEXT            Extension of structure files to be included.
                                  Default = `.pdb`.

  --pieces-filter-file PATH       Path to a Pieces format file used to filter
                                  the dataset to specific chains inspecific
                                  files. All other PDB files included in the
                                  input will be ignored.

  --frame-edge-length FLOAT       Edge length of the cube of space around each
                                  residue that will be voxelized. Default =
                                  12.0 Angstroms.

  --voxels-per-side INTEGER       The number of voxels per side of the frame.
                                  This will give a final cube of `voxels-per-
                                  side`^3. Default = 21.

  -p, --processes INTEGER         Number of processes to be used to create the
                                  dataset. Default = 1.

  -z, --gzipped                   If True, this flag indicates that the
                                  structure files are gzipped. Default =
                                  False.

  -r, --recursive                 If True, all files in all subfolders will be
                                  processed.

  -v, --verbose                   Sets the verbosity of the output, use `-v`
                                  for low level output or `-vv` for even more
                                  information.

  -cb, --encode_cb BOOLEAN        Encode the Cb at an average position
                                  (-0.741287356, -0.53937931, -1.224287356) in
                                  the aligned frame, even for Glycine
                                  residues.

  -ae, --atom_encoder [CNO|CNOCB|CNOCBCA]
                                  Encodes atoms in different channels,
                                  depending on atom types. Default is CNO,
                                  other options are ´CNOCB´ and `CNOCBCA` to
                                  encode the Cb or Cb and Ca in different
                                  channels respectively.  [required]

  --help                          Show this message and exit.

```

## Development

The easiest way to install a development version of `aposteriori` is using
[Poetry](https://python-poetry.org):

```sh
git clone https://github.com/wells-wood-research/aposteriori.git
cd aposteriori/
poetry install
```

In case you get an error about Python Versions (eg. if your python version
 is >3.7), we recommend you use PyEnv for this [PyEnv](https://github.com/pyenv
 /pyenv). 

```sh
pyenv install 3.7.0
pyenv local 3.7.0
```

If you are still having problems switching versions, (eg. you run the previous
 program but running `python -V` shows another version of Python.), try running

```sh
eval "$(pyenv init -)"
```

You can then use either `poetry shell` to activate the development environment or use
`poetry run` to execute single commands in the environment:

```sh
poetry run make-frame-dataset --help
```

Make sure you test your install:

```sh
poetry run pytest tests/
```

if Poetry doesn't pick up on the correct version of Python from PyEnv, run

```sh
poetry env use python3.7
```
