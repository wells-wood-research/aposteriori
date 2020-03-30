import urllib
from pathlib import Path

import ampal
from ampal import AmpalContainer
from ampal.protein import Polypeptide
import src.aposteriori.data_prep.create_data_set as cds
from scipy.stats import entropy
import tensorflow as tf

from src.aposteriori.dnn.config import (
    ANNOTATED_ENTROPY_PDB_PATH,
    PDB_PATH,
    PDB_CODES,
    RADIUS,
    FRAME_CONV_MODEL,
    REBUILD_H5_DATASET,
    PDB_REQUEST_URL,
    FETCH_PDB,
    H5_STRUCTURES_PATH,
    SAVE_ANNOTATED_PDB_TO_FILE,
)
from src.aposteriori.dnn.analysis.callbacks import top_3_cat_acc
from src.aposteriori.dnn.data_processing.discretization import (
    make_data_points,
    FrameDiscretizedProteinsSequence,
)


def _annotate_ampalobj_with_entropy(ampal_structure, prediction_entropy):
    """
    Assigns a B-factor to each residue equivalent to the prediction entropy
    of the model.

    Parameters
    ----------
    ampal_structure : AmpalContainer or Assembly
        Ampal structure to be modified. If an AmpalContainer is passed,
        this will take the first Assembly in the container `ampal_structure[0]`.
    prediction_entropy : numpy.ndarray of floats
        Numpy array with entropy on predictions (n,) where n is the number
        of residues in the structure.

    Returns
    -------
    ampal_structure : Assembly
        Ampal structure with modified B-factor values.
    """
    # Deals with structures from NMR as ampal returns Container of Assemblies
    if isinstance(ampal_structure, AmpalContainer):
        ampal_structure = ampal_structure[0]

    # Reset B-factor:
    for atom in ampal_structure.get_atoms(ligands=True, inc_alt_states=True):
        atom.tags["bfactor"] = 0
    # Apply entropy as B-Factor
    for chain in ampal_structure:
        # Check if chain is Polypeptide (it might be DNA for example...)
        if isinstance(chain, Polypeptide):
            assert len(chain) == len(prediction_entropy), (
                f"Expected a prediction for each residue, but chain is "
                f"{len(chain)} and entropy is {len(prediction_entropy)}"
            )
            for residue, entropy_val in zip(chain, prediction_entropy):
                for atom in residue:
                    atom.tags["bfactor"] = entropy_val

    return ampal_structure


def _fetch_pdb(pdb_code, output_folder=PDB_PATH, pdb_request_url=PDB_REQUEST_URL):
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

    """
    urllib.request.urlretrieve(
        pdb_request_url + pdb_code + ".pdb1.gz",
        filename=output_folder / f"{pdb_code}.pdb1.gz",
    )
    urllib.request.urlretrieve(
        pdb_request_url + pdb_code + ".pdb", filename=output_folder / f"{pdb_code}.pdb",
    )


def _create_paths(pdb_code_list, pdb_path=PDB_PATH, fetch_pdb=FETCH_PDB):
    """
    Creates paths to the PDB files.

    Parameters
    ----------
    pdb_code_list : List of str
        List of pdb codes to be visualized.
    pdb_path : Path object
        Path to pdb folder.
    fetch_pdb : Bool
        Whether PDBs need to be downloaded to a folder.

    Returns
    -------
    pdb_path_list : List of Path object
        List of paths to the pdb files to be analysed.
    """
    print(f"Fetching pdbs from {PDB_PATH}")

    pdb_path_list = []
    for pdb_code in pdb_code_list:

        # Remove format in pdb codes (just in case...)
        if "." in pdb_code:
            pdb_code = pdb_code.split(".")[0]
        # Download PDB:
        if fetch_pdb:
            _fetch_pdb(pdb_code)

        # Append path if structure is available:
        current_path = pdb_path / (pdb_code + ".pdb1.gz")
        (pdb_path_list.append(current_path) if current_path.exists() else None)

    return pdb_path_list


def calculate_prediction_entropy(residue_predictions):
    """
    Calculates Shannon Entropy on predictions.

    Parameters
    ----------
    residue_predictions : numpy.ndarray of floats
        Residue probabilities for each position in sequence of shape (n, 20)
        where n is the number of residues in sequence.

    Returns
    -------
    entropy_arr : numpy.ndarray of floats
        Entropy of prediction for each position in sequence of shape (n,).
    """
    entropy_arr = entropy(residue_predictions, base=2, axis=1)
    return entropy_arr


def visualize_model_entropy(
    model_path=FRAME_CONV_MODEL,
    pdb_codes=PDB_CODES,
    annotated_entropy_pdb_path=ANNOTATED_ENTROPY_PDB_PATH,
    rebuild_h5_dataset=REBUILD_H5_DATASET,
    h5_structures_path=H5_STRUCTURES_PATH,
    radius=RADIUS,
    save_annotated_pdb_to_file=SAVE_ANNOTATED_PDB_TO_FILE,
):
    """
    Visualize Shannon entropy on pdb structures. PDB codes are downloaded
    and predicted by a model specified in `model_path`.

    Parameters
    ----------
    model_path : Path
        Path to aposteriori model.
    pdb_codes : List of str
        List of PDB codes to be analysed.
    annotated_entropy_pdb_path : Path
        Path to save the annotated pdb. Defaults to log file from config.
    rebuild_h5_dataset : Bool
        True will create a .h5 file with the data, while False will try to
        import it from path.
    h5_structures_path : Path
        Path to h5 dataset of structures (or where it should be)
    radius : int
          Length of the edge of the frame unit

                   +--------+
                  /        /|
                 /        / |
                +--------+  |
                |        |  |
                |        |  +
                |        | /
                |        |/
                +--------+
                <-radius->
          (this isn't actually a radius, but it gives the idea)
    save_annotated_pdb_to_file: Bool
        True saves the annotated PDB to a file.

    Returns
    -------
    annotated_pdbs: List of Ampal Assemblies
        List of ampal objects with entropy as B-factor.
    """
    pdb_path_list = _create_paths(pdb_codes)
    # Build an H5 file - No need to rebuild each time, change to
    # REBUILD_H5_DATASET = false if you already ran this.
    if rebuild_h5_dataset:
        # Discretize and load the data:
        cds.process_paths(pdb_path_list, "v1", 4, h5_structures_path)

    # Plot error for each of the PDB codes
    annotated_pdbs = []
    for i in range(len(pdb_codes)):
        test_structures = make_data_points(
            h5_structures_path,
            pdb_codes=[pdb_codes[i] + ".pdb1"],
            radius=radius,
            shuffle=False,
        )
        test_set = FrameDiscretizedProteinsSequence(
            data_set_path=h5_structures_path,
            data_points=test_structures,
            radius=radius,
            batch_size=1,
            shuffle=False,
        )
        tf.keras.utils.get_custom_objects()["top_3_cat_acc"] = top_3_cat_acc
        frame_model = tf.keras.models.load_model(model_path)
        # Make predictions on loaded data:
        final_prediction = frame_model.predict_generator(test_set)
        # Calculate Shannon entropy of predictions:
        entropy_arr = calculate_prediction_entropy(final_prediction)

        # Load PDB Structure:
        current_pdb_path = pdb_path_list[i].parent / (
            pdb_path_list[i].stem[0:4] + ".pdb"
        )
        curr_pdb_structure = ampal.load_pdb(current_pdb_path)

        # Add entropy as B-Factor:
        curr_annotated_structure = _annotate_ampalobj_with_entropy(
            curr_pdb_structure, entropy_arr
        )
        # Add annotated structure to list:
        annotated_pdbs.append(curr_annotated_structure)

        # Save to a file:
        if save_annotated_pdb_to_file:
            curr_pdb_out_filename = annotated_entropy_pdb_path / (
                pdb_path_list[i].stem[0:4] + "_w_entropy.pdb"
            )
            with open(curr_pdb_out_filename, "w") as f:
                f.write(curr_annotated_structure.pdb)

    return annotated_pdbs


if __name__ == "__main__":
    _ = visualize_model_entropy()
