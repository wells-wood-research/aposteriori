import pickle
import sys

from tensorflow.keras.utils import plot_model

from analysis.callbacks import (
    FrameConfusionPlotter,
    top_3_cat_acc,
    csv_logger,
    checkpoint,
    create_tb_callback,
)
from config import *
from data_processing.encoder import encode_data
from data_processing.discretization import FrameDiscretizedProteinsSequence
from network.frame_model import create_frame_cnn_model
from network.visualization.frame import visualize_model_layer
from network.visualization.prediction_entropy import visualize_model_entropy


def log_uncaught_exceptions(ex_type, ex_value, ex_traceback):
    """ Logs unprediceted exceptions from sys.excepthook """
    logging.exception("Uncaught exception",
                      exc_info=(ex_type, ex_value, ex_traceback))


def load_data_from_pickle(data_path):
    with open(data_path, 'rb') as inf:
        imported_data = pickle.load(inf)
    return imported_data


if __name__ == '__main__':
    # Begin logging:
    sys.excepthook = log_uncaught_exceptions
    logger = logging.getLogger(__name__)
    logger.info(f'Starting training at {CURRENT_DATE}\n')
    logger.info(f'Configuration: '
                f'{{Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, '
                f'Radius: {RADIUS}, Shuffle: {SHUFFLE}, '
                f'Loss function: {LOSS_FUNC}, '
                f'Activation function: {ACTIVATION_FUNC}, '
                f'Optimizer: {OPTIMIZER}}}')

    # Retrieve Data:
    TRAINING_DATA = load_data_from_pickle(TRAINING_PATH_FRAME)
    VALIDATION_DATA = load_data_from_pickle(VALIDATION_PATH_FRAME)
    # Discretize Structures:
    TRAINING_SET = FrameDiscretizedProteinsSequence(
        data_set_path=PIECES_DATA_PATH, data_points=TRAINING_DATA[0:200],
        radius=RADIUS, batch_size=BATCH_SIZE, shuffle=SHUFFLE
    )
    VALIDATION_SET = FrameDiscretizedProteinsSequence(
        data_set_path=PIECES_DATA_PATH, data_points=VALIDATION_DATA[0:200],
        radius=RADIUS, batch_size=BATCH_SIZE, shuffle=SHUFFLE
    )
    logger.info(f'Training Set: {len(TRAINING_SET)}, Validation Set: {len(VALIDATION_SET)}')
    assert all(tuple(x[(RADIUS, RADIUS, RADIUS)]) == tuple([0, 1, 0, 0, 0]) for x in TRAINING_SET[0][0])
    assert all(tuple(x[(RADIUS, RADIUS, RADIUS)]) == tuple([0, 1, 0, 0, 0]) for x in TRAINING_SET[0][0])

    # Define Model:
    tensorflow.keras.utils.get_custom_objects()['top_3_cat_acc'] = top_3_cat_acc
    model = create_frame_cnn_model(INPUT_SHAPE_FRAME)
    model.compile(
        loss=LOSS_FUNC,
        optimizer=OPTIMIZER,
        metrics=METRICS
    )
    model.summary(print_fn=lambda x: logging.info(x + '\n'))

    # Architecture Plots:
    try:
        _ = plot_model(model, OUTPUT_DIR / 'model_architecture.svg',
                       show_shapes=True)
    except ValueError:
        pass

    # Define callbacks:
    _, label_encoder = encode_data()
    confusion_plotter = FrameConfusionPlotter(
        OUTPUT_DIR / (NAME_MODEL + '{epoch:02d}-cm.svg'),
        VALIDATION_SET,
        10,
        label_encoder.categories_[0],
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
        visualize_model_layer(-4, VALIDATION_SET, [5, 8])

    # Visualization of the entropy of predictions:
    if VISUALIZE_ENTROPY_AFTER_TRAINING:
        visualize_model_entropy()
