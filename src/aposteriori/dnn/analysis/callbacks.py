import os

import tensorflow.keras as keras
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

from aposteriori.dnn.analysis.metrics import make_frame_confusion_matrix, make_contig_confusion_matrix
from aposteriori.dnn.config import (
    RESIDUES_THREE_TO_ONE_LETTER,
    BLOSUM_DICT,
    OUTPUT_DIR,
    NAME_MODEL,
    TENSORBOARD_OUTPUT_DIR,
    BATCH_SIZE,
    WRITE_GRADS,
    HISTOGRAM_FREQ,
    UPDATE_FREQ,
)


class FrameConfusionPlotter(keras.callbacks.Callback):
    def __init__(self, output_filename, data_gen, batches, encoder_categories):
        super().__init__()
        self.output_filename = str(output_filename)
        self.data_gen = data_gen
        self.batches = batches
        self.encoder_categories = encoder_categories

    def on_epoch_end(self, epoch, logs={}):
        fig = plt.figure()
        plt.imshow(make_frame_confusion_matrix(self.model, self.data_gen, self.batches))
        plt.xlabel("Predicted")
        plt.xticks(range(20), self.encoder_categories, rotation=90)
        plt.ylabel("True")
        plt.yticks(range(20), self.encoder_categories)
        # Plot Color Bar:
        norm = colors.Normalize()
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        fig.colorbar(sm).set_label("Confusion Level (Range 0 - 1)")
        # Save to file:
        plt.savefig(self.output_filename.format(epoch=epoch + 1), bbox_inches="tight")
        plt.close()
        return


class ContigConfusionPlotter(keras.callbacks.Callback):
    def __init__(self, output_filename, data_gen, batches, encoder_categories):
        super().__init__()
        self.output_filename = str(output_filename)
        self.data_gen = data_gen
        self.batches = batches
        self.encoder_categories = encoder_categories

    def on_epoch_end(self, epoch, logs={}):
        fig = plt.figure()
        plt.imshow(
            make_contig_confusion_matrix(self.model, self.data_gen, self.batches)
        )
        plt.xlabel("Predicted")
        plt.xticks(range(20), self.encoder_categories, rotation=90)
        plt.ylabel("True")
        plt.yticks(range(20), self.encoder_categories)
        # Plot Color Bar:
        norm = colors.Normalize()
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        fig.colorbar(sm).set_label("Confusion Level (Range 0 - 1)")
        # Save to file:
        plt.savefig(self.output_filename.format(epoch=epoch + 1), bbox_inches="tight")
        plt.close()
        return


class BlosumScoreCalculator(keras.callbacks.Callback):
    def __init__(self, output_filename, data_gen, batches, encoder_categories):
        super().__init__()
        self.output_filename = str(output_filename)
        self.data_gen = data_gen
        self.batches = batches
        self.encoder_categories = encoder_categories

    def on_epoch_end(self, epoch, logs={}):
        blosum_score_list = []

        for i in range(self.batches):
            # Extract data:
            data_point, labels = self.data_gen[i]

            # Predict residues:
            true_residue = RESIDUES_THREE_TO_ONE_LETTER[
                self.encoder_categories[int(np.argmax(labels, axis=1))]
            ]
            predicted_residue = RESIDUES_THREE_TO_ONE_LETTER[
                self.encoder_categories[
                    int(np.argmax(self.model.predict(data_point), axis=1))
                ]
            ]

            # Calculate blosum score:
            blosum_score = BLOSUM_DICT[(true_residue, predicted_residue)]
            blosum_score_list.append(2 ** (2 * blosum_score))

        # Save avg blosum score to file:
        with open(self.output_filename, "a+") as f:
            epoch = epoch + 1
            f.write(f"{epoch}, {np.mean(blosum_score_list)}\n")

        return


def top_3_cat_acc(y_true, y_pred):
    # TODO: This could take in a custom k
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)


def create_tb_callback(
    log_dir=TENSORBOARD_OUTPUT_DIR,
    histogram_freq=HISTOGRAM_FREQ,
    write_grads=WRITE_GRADS,
    batch_size=BATCH_SIZE,
    update_freq=UPDATE_FREQ,
):
    tb_callback = keras.callbacks.TensorBoard(
        log_dir=str(log_dir),
        histogram_freq=histogram_freq,
        write_grads=write_grads,
        batch_size=batch_size,
        update_freq=update_freq,
    )

    return tb_callback


# Performance to CSV
csv_logger = keras.callbacks.CSVLogger(
    os.path.join(OUTPUT_DIR, f"{NAME_MODEL}.csv"), append=True
)

# Model Checkpoint
checkpoint = keras.callbacks.ModelCheckpoint(
    os.path.join(OUTPUT_DIR, NAME_MODEL + "-{epoch:02d}-{val_loss:.2f}.h5")
)
