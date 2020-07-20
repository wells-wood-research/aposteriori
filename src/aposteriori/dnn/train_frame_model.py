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
from aposteriori.dnn.data_processing.tools import balance_dataset, create_flat_dataset_map, encode_data
from aposteriori.dnn.data_processing.discretization import (
    FrameDiscretizedProteinsSequence,
)
from aposteriori.dnn.network.frame_model import create_frame_2d7_model
from aposteriori.dnn.network.visualization.frame_activation import visualize_model_layer
from aposteriori.dnn.network.visualization.prediction_entropy import (
    visualize_model_entropy,
)


def log_uncaught_exceptions(ex_type, ex_value, ex_traceback):
    """ Logs unpredicted exceptions from sys.excepthook """
    logging.exception("Uncaught exception", exc_info=(ex_type, ex_value, ex_traceback))


if __name__ == "__main__":
    # Begin logging:
    sys.excepthook = log_uncaught_exceptions
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training at {CURRENT_DATE}\n")
    logger.info(
        f"Configuration: "
        f"{{Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, "
        f"Voxels per side: {VOXELS_PER_SIDE}, Shuffle: {SHUFFLE}, "
        f"Loss function: {LOSS_FUNC}, Frame edge length (A): {FRAME_EDGE_LENGTH} "
        f"Activation function: {ACTIVATION_FUNC}, "
        f"Optimizer: {OPTIMIZER}}}"
    )

    flat_dataset_map = create_flat_dataset_map(HDF5_STRUCTURES_PATH)
    flat_dataset_map = balance_dataset(flat_dataset_map)
    # Splitting the dataset
    training_data, validation_data = train_test_split(
        flat_dataset_map, test_size=0.20, random_state=42
    )
    # Discretize Structures:
    TRAINING_SET = FrameDiscretizedProteinsSequence(
        dataset_map=training_data,
        dataset_path=HDF5_STRUCTURES_PATH,
        voxels_per_side=VOXELS_PER_SIDE,
    )
    VALIDATION_SET = FrameDiscretizedProteinsSequence(
        dataset_map=training_data,
        dataset_path=HDF5_STRUCTURES_PATH,
        voxels_per_side=VOXELS_PER_SIDE,
    )
    logger.info(
        f"Training Set: {len(TRAINING_SET)}, Validation Set: {len(VALIDATION_SET)}"
    )

    # Define Model:
    tensorflow.keras.utils.get_custom_objects()["top_3_cat_acc"] = top_3_cat_acc
    model = create_frame_2d7_model(INPUT_SHAPE_FRAME)
    model.compile(loss=LOSS_FUNC, optimizer=OPTIMIZER, metrics=METRICS)
    model.summary(print_fn=lambda x: logging.info(x + "\n"))

    # Architecture Plots:
    try:
        _ = plot_model(model, OUTPUT_DIR / "model_architecture.svg", show_shapes=True)
    except ValueError:
        pass

    # Define callbacks:
    _, residue_encoder = encode_data()
    confusion_plotter = FrameConfusionPlotter(
        OUTPUT_DIR / (NAME_MODEL + "{epoch:02d}-cm.svg"),
        VALIDATION_SET,
        10,
        residue_encoder.categories_[0],
    )
    tb_callback = create_tb_callback()

    # Fit Model
    _ = model.fit_generator(
        generator=TRAINING_SET,
        validation_data=VALIDATION_SET,
        epochs=EPOCHS,
        use_multiprocessing=MULTIPROCESSING,
        workers=WORKERS,
        callbacks=[csv_logger, checkpoint, confusion_plotter, tb_callback],
    )
    # Visualization of the activation layers:
    if VISUALIZE_ACTIVATION_AFTER_TRAINING:
        visualize_model_layer(-4, VALIDATION_SET, [1, 2])

    # Visualization of the entropy of predictions:
    if VISUALIZE_ENTROPY_AFTER_TRAINING:
        visualize_model_entropy()
