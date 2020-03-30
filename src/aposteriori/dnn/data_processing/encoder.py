import pickle
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from aposteriori.dnn.config import ATOMIC_NUMBERS, ENCODER_PATH
from ampal.amino_acids import standard_amino_acids

# TODO: Refactor order as in PEP8


def encode_data():
    # TODO: do not encode if file exist. Add bool to function to optionally
    #  re-encode
    data_encoder = OneHotEncoder(categories="auto", sparse=False,
                                 handle_unknown="ignore")
    data_encoder.fit(np.array(ATOMIC_NUMBERS).reshape(-1, 1))

    label_encoder = OneHotEncoder(categories="auto", sparse=False)
    label_encoder.fit(np.array(list(standard_amino_acids.values())).reshape(-1, 1))

    with open(ENCODER_PATH, "wb") as f:
        pickle.dump({"data_encoder": data_encoder, "label_encoder": label_encoder}, f)

    return data_encoder, label_encoder
