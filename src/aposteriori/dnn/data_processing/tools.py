import copy
import h5py
import sys
import typing as t
import warnings

from ampal.amino_acids import standard_amino_acids
from collections import Counter
from pathlib import Path
from random import shuffle

from aposteriori.data_prep.create_frame_data_set import DatasetMetadata
from aposteriori.dnn.config import UNCOMMON_RESIDUE_DICT, MAKE_FRAME_DATASET_VER


def extract_metadata_from_dataset(frame_dataset: Path) -> DatasetMetadata:
    """
    Retrieves the metadata of the dataset and does a sanity check of the version.
    If the dataset version is not compatible with aposteriori, the training process will stop.

    Parameters
    ----------
    frame_dataset: Path
        Path to the .h5 dataset with the following structure.
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

    Returns
    -------
    dataset_metadata: DatasetMetadata of the dataset with the following parameters:
        make_frame_dataset_ver: str
        frame_dims: t.Tuple[int, int, int, int]
        atom_encoder: t.List[str]
        encode_cb: bool
        atom_filter_fn: str
        residue_encoder: t.List[str]
        frame_edge_length: float

    """
    with h5py.File(frame_dataset, "r") as dataset_file:
        meta_dict = dict(dataset_file.attrs.items())
        dataset_metadata = DatasetMetadata.import_metadata_dict(meta_dict)

    # Extract version metadata:
    dataset_ver_num = dataset_metadata.make_frame_dataset_ver.strip(".")[0]
    aposteriori_ver_num = MAKE_FRAME_DATASET_VER.strip(".")[0]
    # If the versions are compatible, return metadata
    if dataset_ver_num == aposteriori_ver_num:
        return dataset_metadata
    else:
        sys.exit(
            f"Dataset version is {dataset_metadata.make_frame_dataset_ver} and is incompatible "
            f"with Aposteriori version {MAKE_FRAME_DATASET_VER}."
            f"Try re-creating the dataset with the current version of Aposteriori."
        )


def create_flat_dataset_map(frame_dataset: Path) -> t.List[t.Tuple[str, int, str, str]]:
    """
    Flattens the structure of the h5 dataset for batching and balancing
    purposes.

    Parameters
    ----------
    frame_dataset: Path
        Path to the .h5 dataset with the following structure.
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

    Returns
    -------
    flat_dataset_map: t.List[t.Tuple]
        List of tuples with the order
        [... (pdb_code, chain_id, residue_id,  residue_label, encoded_residue) ...]
    """
    standard_residues = list(standard_amino_acids.values())

    with h5py.File(frame_dataset, "r") as dataset_file:
        flat_dataset_map = []
        # Create flattened dataset structure:
        for pdb_code in dataset_file:
            for chain_id in dataset_file[pdb_code].keys():
                for residue_id in dataset_file[pdb_code][chain_id].keys():
                    # Extract residue info:
                    residue_label = dataset_file[pdb_code][chain_id][
                        str(residue_id)
                    ].attrs["label"]

                    if residue_label in standard_residues:
                        pass
                    # If uncommon, attempt conversion of label
                    elif residue_label in UNCOMMON_RESIDUE_DICT.keys():
                        warnings.warn(f"{residue_label} is not a standard residue.")
                        # Convert residue to common residue
                        residue_label = UNCOMMON_RESIDUE_DICT[residue_label]
                        warnings.warn(f"Residue converted to {residue_label}.")
                    else:
                        assert (
                            residue_label in standard_residues
                        ), f"Expected natural amino acid, but got {residue_label}."

                    flat_dataset_map.append(
                        (pdb_code, chain_id, residue_id, residue_label)
                    )

    return flat_dataset_map


def balance_dataset(
    flat_dataset_map: t.List[t.Tuple[str, int, str, str]]
) -> t.List[t.Tuple[str, int, str, str]]:
    """
    Balances the dataset by undersampling the least present residue.

    Parameters
    ----------
    flat_dataset_map: t.List[t.Tuple]
        List of tuples with the order
        [... (pdb_code, chain_id, residue_id,  residue_label) ...]

    Returns
    -------
    balanced_dataset_map: t.List[t.Tuple]
        Balanced list of tuples with the order
        [... (pdb_code, chain_id, residue_id,  residue_label) ...].
        This is balanced by undersampling.
    """
    flat_dataset_map_copy = copy.copy(flat_dataset_map)
    # Randomize appearance of frames
    shuffle(flat_dataset_map_copy)
    # List all resiudes:
    standard_residues = list(standard_amino_acids.values())
    # Extract residues and append to a dictionary using the residue as key:
    dataset_dict = {r: [] for r in standard_residues}

    all_residues_in_dataset = []
    for res_map in flat_dataset_map_copy:
        res = res_map[-1]
        dataset_dict[res].append(res_map)
        all_residues_in_dataset.append(res)
    # Count all residues and calculate the maximum number of residue per class:
    counted_residue_in_dataset = Counter(all_residues_in_dataset)
    # Count how many residues to extract per class:
    max_res_num = counted_residue_in_dataset[
        min(counted_residue_in_dataset, key=counted_residue_in_dataset.get)
    ]
    # Extract residue from dataset:
    balanced_dataset_map = []
    for residue in standard_residues:
        # Extract and append relevant residue:
        balanced_dataset_map += dataset_dict[residue][:max_res_num]
    # Check whether the total number of residues is correct:
    assert (
        len(balanced_dataset_map) == 20 * max_res_num
    ), f"Expected balanced dataset to be {20 * max_res_num} but got {len(balanced_dataset_map)}"
    # Check whether the number of residues per class is correct:
    all_balanced_residues = [res[-1] for res in balanced_dataset_map]
    assert Counter(list(standard_amino_acids.values()) * max_res_num) == Counter(
        all_balanced_residues
    )

    return balanced_dataset_map
