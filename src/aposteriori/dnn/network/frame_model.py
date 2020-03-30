from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv3D,
    Dropout,
    Dense,
    Flatten,
    GlobalAveragePooling3D,
    MaxPooling3D,
    SpatialDropout3D,
    concatenate,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential

from src.aposteriori.dnn.config import ACTIVATION_FUNC


def create_frame_2d7_model(input_shape):

    model = Sequential(
    [Conv3D(128, 4, 1,
         padding='same', kernel_initializer='he_normal', use_bias=False,
         input_shape=input_shape
     ),
     BatchNormalization(),
     Activation("elu"),
     MaxPooling3D(3, 2, padding='same'),
     SpatialDropout3D(0.5),

     Conv3D(256, 3, 1,
         padding='same', kernel_initializer='he_normal', use_bias=False,
     ),
     BatchNormalization(),
     Activation("elu"),
     MaxPooling3D(3, 2, padding='same'),

     Conv3D(348, 2, 1,
         padding='same', kernel_initializer='he_normal', use_bias=False,
     ),
     BatchNormalization(),
     Activation("elu"),
     Conv3D(348, 2, 1,
         padding='same', kernel_initializer='he_normal', use_bias=False,
     ),
     BatchNormalization(),
     Activation("elu"),
     Conv3D(256, 2, 1,
         padding='same', kernel_initializer='he_normal', use_bias=False,
     ),
     BatchNormalization(),
     Activation("elu"),

     Conv3D(20, 1, 1,
         padding='same', kernel_initializer='he_normal', use_bias=False,
     ),
     BatchNormalization(),
     Activation("elu"),
     SpatialDropout3D(0.5),

     GlobalAveragePooling3D(),
     Activation("softmax"),
    ])
    return model


def create_frame_cnn_model(input_shape):
    """
        :return: Keras model object
    """

    input_cube = Input(shape=(input_shape))

    tower_1 = Conv3D(16, 2, 1, padding="same", activation=ACTIVATION_FUNC)(input_cube)
    tower_2 = Conv3D(32, 3, 1, padding="same", activation=ACTIVATION_FUNC)(input_cube)
    tower_3 = Conv3D(32, 4, 1, padding="same", activation=ACTIVATION_FUNC)(input_cube)

    layer_x = concatenate([tower_1, tower_2, tower_3], axis=4)
    layer_x = MaxPooling3D((2, 2, 2))(layer_x)

    layer_2 = Conv3D(64, 3, 1, padding="same", activation=ACTIVATION_FUNC)(layer_x)
    layer_2 = MaxPooling3D((3, 3, 3))(layer_2)
    layer_2 = Flatten()(layer_2)
    layer_2 = Dense(1024, activation="softmax")(layer_2)

    layer_3 = Dropout(0.2)(layer_2)
    end_tower = Dense(20, kernel_initializer="he_normal", activation="softmax")(layer_3)

    model = Model([input_cube], end_tower)

    return model


def create_frame_cnn_model2019(input_shape):
    """
        :return: Keras model object
    """

    input_cube = Input(shape=(input_shape))

    tower_1 = Conv3D(16, 2, 1, padding="same", activation="relu")(input_cube)

    tower_2 = Conv3D(32, 3, 1, padding="same", activation="relu")(input_cube)

    tower_3 = Conv3D(32, 4, 1, padding="same", activation="relu")(input_cube)

    layer_x = concatenate([tower_1, tower_2, tower_3], axis=4)

    layer_x = MaxPooling3D((2, 2, 2))(layer_x)

    layer_2 = Conv3D(64, 3, 1, padding="same", activation="relu")(layer_x)

    layer_2 = MaxPooling3D((3, 3, 3))(layer_2)

    layer_2 = Flatten()(layer_2)

    layer_2 = Dense(1024, activation="softmax")(layer_2)

    layer_3 = Dropout(0.2)(layer_2)

    model = Model([input_cube], layer_3)

    print(model.summary())

    return model
