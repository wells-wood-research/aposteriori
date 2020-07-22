import pickle
import h5py
import numpy as np
import typing as t
from operator import itemgetter
import warnings
import copy

from ampal.amino_acids import standard_amino_acids
from collections import Counter
from pathlib import Path
from random import shuffle
from sklearn.preprocessing import OneHotEncoder

from aposteriori.dnn.config import ATOMIC_NUMBERS, ENCODER_PATH, UNCOMMON_RESIDUE_DICT


def encode_data(encoder_path: Path = ENCODER_PATH, load_existing_encoder: bool = True):
    """

    Parameters
    ----------
    encoder_path: Path
        Path to the encoder. If it exists it loads it for use.
    load_existing_encoder: bool
        Whether to load an existing encoder or recreate it.

    Returns
    -------
    atom_encoder: sklearn.preprocessing._encoders.OneHotEncoder
        Encodes atoms in frame to one-hot encoded atoms.

    residue_encoder: sklearn.preprocessing._encoders.OneHotEncoder
        Encodes residues label to one-hot encoded residues. Uses ampal standard
        residues as guide.
    """

    if encoder_path.exists() and load_existing_encoder:
        with open(encoder_path, "rb") as f:
            encoders = pickle.load(f)
            atom_encoder = encoders["atom_encoder"]
            residue_encoder = encoders["residue_encoder"]

    else:
        atom_encoder = OneHotEncoder(
            categories="auto", sparse=False, handle_unknown="ignore"
        )
        atom_encoder.fit(np.array(ATOMIC_NUMBERS).reshape(-1, 1))

        residue_encoder = OneHotEncoder(categories="auto", sparse=False)
        residue_encoder.fit(
            np.array(list(standard_amino_acids.values())).reshape(-1, 1)
        )
        with open(encoder_path, "wb") as f:
            pickle.dump(
                {"atom_encoder": atom_encoder, "residue_encoder": residue_encoder}, f
            )

    return atom_encoder, residue_encoder


def create_flat_dataset_map(frame_dataset: Path):
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

    Returns
    -------

    flat_dataset_map: t.List[t.Tuple]
        List of tuples with the order
        [... (pdb_code, chain_id, residue_id,  residue_label) ...]
    """
    standard_residues = list(standard_amino_acids.values())

    with h5py.File(frame_dataset, "r") as dataset_file:
        flat_dataset_map = []
        # Create flattened dataset structure:
        for pdb_code in dataset_file:
            for chain_id in dataset_file[pdb_code].keys():
                for residue_id in dataset_file[pdb_code][chain_id].keys():
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


def balance_dataset(flat_dataset_map: t.List[t.Tuple]):
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
    dataset_dict = {r:[] for r in standard_residues}

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
    assert (Counter(list(standard_amino_acids.values())*max_res_num) == Counter(all_balanced_residues))

    return balanced_dataset_map
