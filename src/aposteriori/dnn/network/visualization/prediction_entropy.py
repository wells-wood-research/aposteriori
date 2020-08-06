import urllib
import typing as t
import warnings
from pathlib import Path

import numpy as np
import ampal
import tensorflow as tf
from ampal.protein import Polypeptide
from scipy.stats import entropy

from aposteriori.dnn.config import (
    ANNOTATED_ENTROPY_PDB_PATH,
    FRAME_EDGE_LENGTH,
    PDB_PATH,
    PDB_CODES,
    FRAME_CONV_MODEL,
    PDB_REQUEST_URL,
    SAVE_ANNOTATED_PDB_TO_FILE,
    VOXELS_PER_SIDE,
)
from aposteriori.dnn.network.analysis.callbacks import top_3_cat_acc
from aposteriori.dnn.data_processing.discretization import (
    FrameDiscretizedProteinsSequence,
)
from aposteriori.data_prep.create_frame_data_set import make_frame_dataset
from aposteriori.dnn.data_processing.tools import create_flat_dataset_map


def _annotate_ampalobj_with_entropy(
    ampal_structure: ampal.Assembly, prediction_entropy: np.ndarray
) -> ampal.assembly:
    """
    Assigns a B-factor to each residue equivalent to the prediction entropy
    of the model.

    Parameters
    ----------
    ampal_structure : ampal.Assembly or ampal.AmpalContainer
        Ampal structure to be modified. If an ampal.AmpalContainer is passed,
        this will take the first Assembly in the ampal.AmpalContainer `ampal_structure[0]`.
    prediction_entropy : numpy.ndarray of floats
        Numpy array with entropy on predictions (n,) where n is the number
        of residues in the structure.

    Returns
    -------
    ampal_structure : Assembly
        Ampal structure with modified B-factor values.
    """
    # Deals with structures from NMR as ampal returns Container of Assemblies
    if isinstance(ampal_structure, ampal.AmpalContainer):
        warnings.warn(f"Selecting the first state from the NMR structure {ampal_structure.id}")
        ampal_structure = ampal_structure[0]

    # Reset B-factor:
    for atom in ampal_structure.get_atoms(ligands=True, inc_alt_states=True):
        atom.tags["bfactor"] = 0

    total_length = 0
    # Apply entropy as B-Factor to each chain
    for i, chain in enumerate(ampal_structure):
        # Check if chain is Polypeptide (it might be DNA for example...)
        if isinstance(chain, Polypeptide):
            total_length += len(chain)
            for residue, entropy_val in zip(
                chain, prediction_entropy[total_length : len(chain)]
            ):
                for atom in residue:
                    atom.tags["bfactor"] = entropy_val
    # Check whether the residues in chains were all covered by a posteriori
    assert total_length == len(prediction_entropy), (
        f"Expected a prediction for each residue, but chain is "
        f"{len(total_length)} and entropy is {len(prediction_entropy)}"
    )
    return ampal_structure


def _fetch_pdb(
    pdb_code: str,
    output_folder: Path = PDB_PATH,
    pdb_request_url: str = PDB_REQUEST_URL,
    download_assembly: bool = False,
) -> Path:
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
        Whether to download the assembly file of the pdb.

    Returns
    -------
    output_path: Path
        Path to downloaded pdb

    """
    if download_assembly:
        pdb_code_with_extension = f"{pdb_code}.pdb1"
    else:
        pdb_code_with_extension = f"{pdb_code}.pdb"

    output_path = output_folder / pdb_code_with_extension
    urllib.request.urlretrieve(
        pdb_request_url + pdb_code_with_extension, filename=output_path,
    )

    return output_path


def calculate_prediction_entropy(residue_predictions: list) -> list:
    """
    Calculates Shannon Entropy on predictions.

    Parameters
    ----------
    residue_predictions: list[float]
        Residue probabilities for each position in sequence of shape (n, 20)
        where n is the number of residues in sequence.

    Returns
    -------
    entropy_arr: list[float]
        Entropy of prediction for each position in sequence of shape (n,).
    """
    entropy_arr = entropy(residue_predictions, base=2, axis=1)
    return entropy_arr


def visualize_model_entropy(
    model_path: Path = FRAME_CONV_MODEL,
    pdb_codes: list = PDB_CODES,
    save_annotated_pdb_to_file: bool = SAVE_ANNOTATED_PDB_TO_FILE,
    output_folder: Path = ANNOTATED_ENTROPY_PDB_PATH,
    voxels_per_side: int = VOXELS_PER_SIDE,
    frame_edge_length: int = FRAME_EDGE_LENGTH,
) -> t.List[ampal.Assembly]:
    """
    Visualize Shannon entropy on pdb structures. PDB codes are downloaded
    and predicted by a model specified in `model_path`.

    Parameters
    ----------
    model_path: Path
        Path to aposteriori model.
    pdb_codes: List of str
        List of PDB codes to be analysed.
    save_annotated_pdb_to_file: bool
        Whether to save the annotated pdb to file.
    output_folder: Path
        Path to folder to save the pdb file
    voxels_per_side: int
        Number of voxels per side
    frame_edge_length: int
        Length of the edge of the frame unit in Angstroms.

               +--------+
              /        /|
             /        / |
            +--------+  |
            |        |  |
            |        |  +
            |        | /
            |        |/
            +--------+
            <- this ->

    Returns
    -------
    annotated_pdbs: t.list[ampal.Assembly]
        Lists of annotated ampal assemblies.
    """
    pdb_paths = [_fetch_pdb(pdb_code) for pdb_code in pdb_codes]

    annotated_pdbs = []
    for pdb_path in pdb_paths:
        # Voxelise pdb:
        voxelised_dataset = make_frame_dataset(
            structure_files=[pdb_path],
            output_folder=output_folder,
            name=str(pdb_path),
            voxels_per_side=voxels_per_side,
            frame_edge_length=frame_edge_length,
        )

        flat_dataset_map = create_flat_dataset_map(voxelised_dataset)

        discretized_dataset = FrameDiscretizedProteinsSequence(
            dataset_map=flat_dataset_map,
            dataset_path=voxelised_dataset,
            voxels_per_side=voxels_per_side,
            shuffle=False,
            batch_size=1,
        )
        # Import model:
        tf.keras.utils.get_custom_objects()["top_3_cat_acc"] = top_3_cat_acc
        frame_model = tf.keras.models.load_model(model_path)
        # Make predictions on loaded data:
        final_prediction = frame_model.predict_generator(discretized_dataset)
        # Calculate Shannon entropy of predictions:
        entropy_arr = calculate_prediction_entropy(final_prediction)
        # Plot error for each of the PDB codes:
        curr_pdb_structure = ampal.load_pdb(pdb_path)
        # Add entropy as B-Factor:
        curr_annotated_structure = _annotate_ampalobj_with_entropy(
            curr_pdb_structure, entropy_arr
        )
        # Add annotated structure to list:
        annotated_pdbs.append(curr_annotated_structure)
        # Save to a file:
        if save_annotated_pdb_to_file:
            curr_pdb_filename = output_folder / (
                pdb_path.with_suffix("").stem + "_w_entropy.pdb"
            )
            with open(curr_pdb_filename, "w") as f:
                f.write(curr_annotated_structure.pdb)

    return annotated_pdbs


if __name__ == "__main__":
    _ = visualize_model_entropy()
