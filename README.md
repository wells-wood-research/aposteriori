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
make-frame-dataset -o /tmp/ -n test_dataset pdb_files/**/*.pdb
```

Check the `make-frame-dataset` help page for more details on its usage:

```sh
Usage: make-frame-dataset [OPTIONS] [STRUCTURE_FILES]...

  Creates a dataset of voxelized amino acid frames.

  A frame refers to a region of space around an amino acid. For every
  residue in the input structure(s), a cube of space around the region (with
  an edge length equal to `--frame_edge_length`, default 12 Å), will be
  mapped to discrete space, with a defined number of voxels per edge (equal
  to `--voxels-per-side`, default = 21).

  Basic Usage:

  `make-frame-dataset -o /tmp/ -n test_dataset 1ubq.pdb 1ctf.pdb`

  This command will make a tiny dataset found at `/tmp/test_dataset.hdf5`,
  containing all residues 1ubq.pdb and 1ctf.pdb

  Globs can be used to define the structure files to be processed. `make-
  frame-dataset pdb_files/**/*.pdb` would include all `.pdb` files in all
  subdirectories of the `pdb_files` directory.

Options:
  -o, --output-folder PATH   Path to folder where output will be written.
  -n, --name TEXT            Name used for the dataset file, the `.hdf5`
                             extension does not need to be included as it will
                             be appended.

  --frame-edge-length FLOAT  Edge length of the cube of space around each
                             residue that will be voxelized.

  --voxels-per-side INTEGER  The number of voxels per side of the frame. This
                             will give a final cube of `voxels-per-side`^3.

  -v, --verbose              Sets the verbosity of the output, use `-v` for
                             low level output or `-vv` for even more
                             information.

  --help                     Show this message and exit.
```

## Development

The easiest way to install a development version of `aposteriori` is using
[Poetry](https://python-poetry.org):

```sh
git clone https://github.com/wells-wood-research/aposteriori.git
cd aposteriori/
poetry install
```

In case you get an error about Python Versions (eg. if your python version is >3.7), we suggest you use pyenv. 

```sh
pyenv install 3.7.0
pyenv local 3.7.0
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
