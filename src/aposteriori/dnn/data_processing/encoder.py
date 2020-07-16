import pickle
import numpy as np

from pathlib import Path

from ampal.amino_acids import standard_amino_acids
from sklearn.preprocessing import OneHotEncoder

from aposteriori.dnn.config import ATOMIC_NUMBERS, ENCODER_PATH


def encode_data(encoder_path: Path = ENCODER_PATH, load_existing_encoder: bool = True):
    """

    Parameters
    ----------
    encoder_path: Path
        Path to the encoder. If it exists it loads it for use.
    load_existing_encoder: bool
        Whether to load an existing encoder or recreate it.

    Returns
    -------
    atom_encoder: sklearn.preprocessing._encoders.OneHotEncoder
        Encodes atoms in frame to one-hot encoded atoms.

    residue_encoder: sklearn.preprocessing._encoders.OneHotEncoder
        Encodes residues label to one-hot encoded residues. Uses ampal standard
        residues as guide.
    """

    if encoder_path.exists() and load_existing_encoder:
        with open(encoder_path, "rb") as f:
            encoders = pickle.load(f)
            atom_encoder = encoders["atom_encoder"]
            residue_encoder = encoders["residue_encoder"]

    else:
        atom_encoder = OneHotEncoder(
            categories="auto", sparse=False, handle_unknown="ignore"
        )
        atom_encoder.fit(np.array(ATOMIC_NUMBERS).reshape(-1, 1))

        residue_encoder = OneHotEncoder(categories="auto", sparse=False)
        residue_encoder.fit(
            np.array(list(standard_amino_acids.values())).reshape(-1, 1)
        )

        with open(encoder_path, "wb") as f:
            pickle.dump(
                {"atom_encoder": atom_encoder, "residue_encoder": residue_encoder}, f
            )

    return atom_encoder, residue_encoder
