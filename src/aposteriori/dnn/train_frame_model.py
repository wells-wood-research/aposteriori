import sys

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model

from aposteriori.dnn.network.analysis.callbacks import (
    FrameConfusionPlotter,
    top_3_cat_acc,
    csv_logger,
    checkpoint,
    create_tb_callback,
)
from aposteriori.dnn.config import *
from aposteriori.dnn.data_processing.tools import (
    balance_dataset,
    create_flat_dataset_map,
    extract_metadata_from_dataset,
)
from aposteriori.dnn.data_processing.discretization import (
    FrameDiscretizedProteinsSequence,
)
from aposteriori.dnn.network.frame_model import create_frame_2d7_model
from aposteriori.dnn.network.visualization.frame_activation import visualize_model_layer
from aposteriori.dnn.network.visualization.prediction_entropy import (
    visualize_model_entropy,
)
from ampal.amino_acids import standard_amino_acids


def log_uncaught_exceptions(ex_type, ex_value, ex_traceback):
    """ Logs unpredicted exceptions from sys.excepthook """
    logging.exception("Uncaught exception", exc_info=(ex_type, ex_value, ex_traceback))


if __name__ == "__main__":
    # Begin logging:
    sys.excepthook = log_uncaught_exceptions
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training at {CURRENT_DATE}\n")
    # Extract Metadata + Check compatibility of dataset:
    dataset_metadata = extract_metadata_from_dataset(HDF5_STRUCTURES_PATH)
    logger.info(
        f"Configuration: "
        f"{{Dataset Metadata: {str(dataset_metadata)}, "
        f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, "
        f"Shuffle: {SHUFFLE}, Loss function: {LOSS_FUNC}, "
        f"Activation function: {ACTIVATION_FUNC}, "
        f"Optimizer: {OPTIMIZER}}} "
    )
    # Flattens data structure for balancing / simpler processing.
    flat_dataset_map = create_flat_dataset_map(HDF5_STRUCTURES_PATH)
    logger.info(f"Started with {len(flat_dataset_map)} frames.\n")
    if BALANCE_RESIDUES:
        flat_dataset_map = balance_dataset(flat_dataset_map)
        logger.info(f"Balanced to {len(flat_dataset_map)} frames.\n")
    # Splitting the dataset
    training_data, validation_data = train_test_split(
        flat_dataset_map, test_size=0.20, random_state=42
    )
    logger.info(
        f"Training Set: {len(training_data)}, Validation Set: {len(validation_data)}"
    )
    # Discretize Structures:
    TRAINING_SET = FrameDiscretizedProteinsSequence(
        dataset_map=training_data,
        dataset_path=HDF5_STRUCTURES_PATH,
        batch_size=BATCH_SIZE,
    )
    VALIDATION_SET = FrameDiscretizedProteinsSequence(
        dataset_map=validation_data,
        dataset_path=HDF5_STRUCTURES_PATH,
        batch_size=BATCH_SIZE,
    )
    logger.info(
        f"Batched Training Set: {len(TRAINING_SET)}, Batched Validation Set: {len(VALIDATION_SET)}"
    )

    # Define Model:
    tensorflow.keras.utils.get_custom_objects()["top_3_cat_acc"] = top_3_cat_acc
    model = create_frame_2d7_model(dataset_metadata.frame_dims)
    model.compile(loss=LOSS_FUNC, optimizer=OPTIMIZER, metrics=METRICS)
    model.summary(print_fn=lambda x: logging.info(x + "\n"))

    # Architecture Plots:
    try:
        _ = plot_model(
            model, str(OUTPUT_DIR / "model_architecture.svg"), show_shapes=True
        )
    except ValueError:
        pass

    # Define callbacks:
    confusion_plotter = FrameConfusionPlotter(
        OUTPUT_DIR / (NAME_MODEL + "{epoch:02d}-cm.svg"),
        VALIDATION_SET,
        10,
        list(standard_amino_acids.values()),
    )
    tb_callback = create_tb_callback()

    # # Fit Model
    # _ = model.fit(
    #     x=TRAINING_SET,
    #     validation_data=VALIDATION_SET,
    #     epochs=EPOCHS,
    #     use_multiprocessing=MULTIPROCESSING,
    #     workers=WORKERS,
    #     callbacks=[csv_logger, checkpoint, confusion_plotter, tb_callback],
    # )
    # Visualization of the activation layers:
    # if VISUALIZE_ACTIVATION_AFTER_TRAINING:
    #     visualize_model_layer(-4, VALIDATION_SET, [1, 2])

    # Visualization of the entropy of predictions:
    if VISUALIZE_ENTROPY_AFTER_TRAINING:
        visualize_model_entropy(dataset_metadata)
