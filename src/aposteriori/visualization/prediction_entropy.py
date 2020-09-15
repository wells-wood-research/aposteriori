import typing as t
import warnings
from pathlib import Path

import numpy as np
import ampal
import tensorflow as tf
from ampal.protein import Polypeptide
from scipy.stats import entropy

from aposteriori.data_prep.create_frame_data_set import _fetch_pdb
from aposteriori.config import (
    ANNOTATED_ENTROPY_PDB_PATH,
    DATA_FOLDER,
    PDB_CODES,
    FRAME_CONV_MODEL,
    SAVE_ANNOTATED_PDB_TO_FILE,
    PDB_PATH,
)
from aposteriori.dnn.network.analysis.callbacks import top_3_cat_acc
from aposteriori.dnn.data_processing.discretization import (
    FrameDiscretizedProteinsSequence,
)
from aposteriori.data_prep.create_frame_data_set import (
    make_frame_dataset,
    default_atom_filter,
    encode_residue,
    Codec,
    DatasetMetadata,
)
from aposteriori.dnn.data_processing.tools import create_flat_dataset_map


def _annotate_ampalobj_with_data_tag(
    ampal_structure: ampal.Assembly,
    data_to_annotate: t.List[t.Union[t.List[float], t.List[float]]],
    tags: t.List[str],
) -> ampal.assembly:
    """
    Assigns a data point to each residue equivalent to the prediction the
    tag value. The original value of the tag will be reset to the minimum value
    to allow for a more realistic color comparison.

    Parameters
    ----------
    ampal_structure : ampal.Assembly or ampal.AmpalContainer
        Ampal structure to be modified. If an ampal.AmpalContainer is passed,
        this will take the first Assembly in the ampal.AmpalContainer `ampal_structure[0]`.
    data_to_annotate : numpy.ndarray of numpy.ndarray of floats
        Numpy array with data points to annotate (x, n) where x is the
        numer of arrays with data points (eg, [ entropy, accuracy ] ,
        x = 2n) and n is the number of residues in the structure.
    tags : t.List[str]
        List of string tags of the pdb object (eg. "b-factor")

    Returns
    -------
    ampal_structure : Assembly
        Ampal structure with modified B-factor values.
    """
    assert len(tags) == len(
        data_to_annotate
    ), "The number of tags to annotate and the type of data to annotate have different lengths."
    # Deals with structures from NMR as ampal returns Container of Assemblies
    if isinstance(ampal_structure, ampal.AmpalContainer):
        warnings.warn(
            f"Selecting the first state from the NMR structure {ampal_structure.id}"
        )
        ampal_structure = ampal_structure[0]

    if len(data_to_annotate) > 1:
        assert len(data_to_annotate[0]) == len(data_to_annotate[1]), (
            f"Data to annotatate has shape {len(data_to_annotate[0])} and "
            f"{len(data_to_annotate[1])}. They should be the same."
        )

    for i, tag in enumerate(tags):
        # Reset existing values:
        for atom in ampal_structure.get_atoms(ligands=True, inc_alt_states=True):
            atom.tags[tag] = np.min(data_to_annotate[i])

    # Apply data as tag:
    for chain in ampal_structure:
        for i, tag in enumerate(tags):

            # Check if chain is Polypeptide (it might be DNA for example...)
            if isinstance(chain, Polypeptide):
                assert len(chain) == len(data_to_annotate[i]), (
                    f"Expected a prediction for each residue, but chain is "
                    f"{len(chain)} and entropy is {len(data_to_annotate[i])}"
                )
                for residue, data_val in zip(chain, data_to_annotate[i]):
                    for atom in residue:
                        atom.tags[tag] = data_val

    return ampal_structure


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
    dataset_metadata: DatasetMetadata,
    model_path: Path = FRAME_CONV_MODEL,
    pdb_codes: list = PDB_CODES,
    save_annotated_pdb_to_file: bool = SAVE_ANNOTATED_PDB_TO_FILE,
    output_folder: Path = ANNOTATED_ENTROPY_PDB_PATH,
) -> t.List[ampal.Assembly]:
    """
    Visualize Shannon entropy on pdb structures. PDB codes are downloaded
    and predicted by a model specified in `model_path`.

    Parameters
    ----------
    dataset_metadata: DatasetMetadata
        Metadata of the Dataset used to generate the model.
    model_path: Path
        Path to aposteriori model.
    pdb_codes: List of str
        List of PDB codes to be analysed.
    save_annotated_pdb_to_file: bool
        Whether to save the annotated pdb to file.
    output_folder: Path
        Path to folder to save the pdb file

    Returns
    -------
    annotated_pdbs: t.list[ampal.Assembly]
        Lists of annotated ampal assemblies.
    """
    pdb_paths = [_fetch_pdb(pdb_code) for pdb_code in pdb_codes]
    # Set up codec:
    if set(dataset_metadata.atom_encoder) == {"C", "N", "O"}:
        codec = Codec.CNO()
    elif set(dataset_metadata.atom_encoder) == {"C", "N", "O", "CB"}:
        codec = Codec.CNOCB()
    elif set(dataset_metadata.atom_encoder) == {"C", "N", "O", "CB", "CA"}:
        codec = Codec.CNOCBCA()

    annotated_pdbs = []
    for pdb_path in pdb_paths:
        # Voxelise pdb:
        voxelised_dataset = make_frame_dataset(
            structure_files=[pdb_path],
            output_folder=output_folder,
            name=str(pdb_path),
            voxels_per_side=dataset_metadata.frame_dims[0],
            frame_edge_length=dataset_metadata.frame_edge_length,
            codec=codec,
            require_confirmation=False,
        )

        flat_dataset_map = create_flat_dataset_map(voxelised_dataset)

        discretized_dataset = FrameDiscretizedProteinsSequence(
            dataset_map=flat_dataset_map,
            dataset_path=voxelised_dataset,
            shuffle=False,
            batch_size=1,
        )
        # Import model:
        tf.keras.utils.get_custom_objects()["top_3_cat_acc"] = top_3_cat_acc
        frame_model = tf.keras.models.load_model(model_path)
        # Make predictions on loaded data:
        final_prediction = frame_model.predict(discretized_dataset)
        # Calculate Shannon entropy of predictions:
        entropy_arr = calculate_prediction_entropy(final_prediction)
        # Extract probability for real residue at position:
        residue_labels_map = np.asarray(
            [encode_residue(label[3]) for label in flat_dataset_map]
        )
        probability_real_res_arr = final_prediction[residue_labels_map > 0]
        # Plot error for each of the PDB codes:
        curr_pdb_structure = ampal.load_pdb(pdb_path)
        # Add entropy as B-Factor and prediction accuracy as Occupancy:
        curr_annotated_structure = _annotate_ampalobj_with_data_tag(
            curr_pdb_structure,
            [entropy_arr, probability_real_res_arr],
            tags=["bfactor", "occupancy"],
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


def evaluate_model_accuracy(
    model_path: Path = FRAME_CONV_MODEL,
    pdb_codes: t.List[str] = PDB_CODES,
    rebuild_h5_dataset: bool = True,
    pdb_outpath: Path = PDB_PATH,
    dataset_output_path: Path = DATA_FOLDER,
    atom_encoder: str = "CNO",
    name: str = "model_evaluation",
    frame_edge_length: float = 12.0,
    voxels_per_side: int = 21,
    atom_filter_fn: t.Callable = default_atom_filter,
    pieces_filter_file: t.Optional[Path] = None,
    processes: int = 1,
    gzipped: bool = False,
    encode_cb: bool = True,
):
    """
    Evaluate the accuracy of a model on a set of pdb structures.

    Parameters
    ----------
    model_path: Path
        Path to model to evaluate
    pdb_codes: t.List[str]
        List of PDB codes to test with the model. Default are "1qys" and "6ct4"
    rebuild_h5_dataset: bool
        Whether the dataset should be rebuilt. Default = True. Use False if you
         already ran this once before, it might save some time.
    pdb_outpath: Path
        Path to where the PDB structures will be saved. Default is DATA_FOLDER/pdb/
    dataset_output_path: Path
        Path to where the .h5 dataset will be saved. Default is DATA_FOLDER.
    atom_encoder: str
        Encoders used for voxelisation. Default is "CNO".
    name: str
        Name given to the .hdf5 file. Default is "model_evaluation",
    frame_edge_length: float
        Edge length of the cube of space around each residue that will be voxelized.
        Default = 12.0 Angstroms.
    voxels_per_side: int
        The number of voxels per side of the frame. This will give a final cube of `voxels-per-side`^3. Default = 21.
    atom_filter_fn: t.Callable
        Function to filter atoms. Default = default_atom_filter.
    pieces_filter_file: t.Optional[Path]
        Path to a Pieces format file used to filter the dataset to specific chains in
        specific files. All other PDB files included in the input will be ignored.
    processes: int
        Number of processes to be used to create the dataset. Default = 1.
    gzipped: bool
        If True, this flag indicates that the structure files are gzipped. Default = False,
    encode_cb: bool
        Encode the Cb at an average position (-0.741287356, -0.53937931, -1.224287356) in the aligned frame, even for Glycine residues. Default = True,

    Returns
    -------
    final_prediction: List of float
        List of loss, accuracy and top-3 accuracy
    """
    pdb_path_list = [
        _fetch_pdb(pdb_code, download_assembly=True, output_folder=pdb_outpath)
        for pdb_code in pdb_codes
    ]

    # Build an H5 file - No need to rebuild each time, change to
    # rebuild_h5_dataset = False if you already ran this.
    if rebuild_h5_dataset:
        # Create Codec:
        if atom_encoder == "CNO":
            codec = Codec.CNO()
        elif atom_encoder == "CNOCB":
            codec = Codec.CNOCB()
        elif atom_encoder == "CNOCBCA":
            codec = Codec.CNOCBCA()
        else:
            assert atom_encoder in [
                "CNO",
                "CNOCB",
                "CNOCBCA",
            ], f"Expected encoder to be CNO, CNOCB, CNOCBCA but got {atom_encoder}"

        # Voxelise the data:
        voxelised_dataset = make_frame_dataset(
            structure_files=pdb_path_list,
            output_folder=dataset_output_path,
            name=name,
            frame_edge_length=frame_edge_length,
            voxels_per_side=voxels_per_side,
            codec=codec,
            atom_filter_fn=atom_filter_fn,
            pieces_filter_file=pieces_filter_file,
            processes=processes,
            gzipped=gzipped,
            encode_cb=encode_cb,
            require_confirmation=False,
        )
    else:
        voxelised_dataset = dataset_output_path / name
        assert (
            voxelised_dataset.exists()
        ), f"Dataset file at location {voxelised_dataset} not found."

    flat_dataset_map = create_flat_dataset_map(voxelised_dataset)
    discretized_dataset = FrameDiscretizedProteinsSequence(
        dataset_map=flat_dataset_map,
        dataset_path=voxelised_dataset,
        shuffle=False,
        batch_size=1,
    )
    # Predict:
    tf.keras.utils.get_custom_objects()["top_3_cat_acc"] = top_3_cat_acc
    frame_model = tf.keras.models.load_model(model_path)
    final_prediction = frame_model.evaluate(discretized_dataset)

    return final_prediction


if __name__ == "__main__":
    evaluate_model_accuracy()
