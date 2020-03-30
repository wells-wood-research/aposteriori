from tensorflow.keras.layers import (
    Dense,
    Flatten,
    TimeDistributed,
    Conv1D,
    BatchNormalization,
    Activation,
    GlobalAveragePooling1D,
)

from tensorflow.keras.models import Model, load_model
from tensorflow.keras import Sequential
from aposteriori.dnn.analysis.callbacks import top_3_cat_acc
from tensorflow.keras import utils

from aposteriori.dnn.config import FRAME_CONV_MODEL


def create_contig_rnn_model(input_shape):

    utils.get_custom_objects()['top_3_cat_acc'] = top_3_cat_acc
    frame_model = load_model(FRAME_CONV_MODEL)

    model = Sequential(
        [
            TimeDistributed(frame_model, input_shape=input_shape),
            Flatten(),
            Dense(20, activation="sigmoid"),
        ]
    )
    print(model.summary())

    return model

def create_contig_cnn_model(input_shape):

    model = Sequential(
        [
            Conv1D(32, 3, padding="same", input_shape=input_shape),
            BatchNormalization(),
            Activation("elu"),
            Conv1D(20, 1, padding="same"),
            BatchNormalization(),
            Activation("elu"),
            Conv1D(20, 1, padding="same"),
            GlobalAveragePooling1D(),
            Activation("softmax"),
        ]
    )
    print(model.summary())

    return model