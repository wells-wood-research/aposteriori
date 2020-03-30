import h5py
import random

import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from ampal.amino_acids import standard_amino_acids

from src.aposteriori.dnn.data_processing.encoder import encode_data
from src.aposteriori.dnn.config import UNCOMMON_RESIDUE_DICT, UNCOMMON_RES_CONVERSION


class FrameDiscretizedProteinsSequence(keras.utils.Sequence):
    """
      Keras Sequence object containing protein frames.

      Each of the protein frames is voxelised using the label encoder. The
      alpha carbons are centered within these frames.

      Extra padding is added to the frame to obtain a frame of specified
      radius.


      Attributes
      ----------
      data_set_path : Path Object or str
          Path to structural data

      data_points : array of tuples
          [(PDB Code, [Ca coordinates], residue identity of Ca, atom encoder)]

          eg.
          [
          '5c7h.pdb1',
          (55, 61, 41),
          'LYS',
           array([0, 6, 7, 8])
           ...]

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

      batch_size : int
          Number of data_points considered at once

      shuffle : bool
          Shuffling of the order of data_points


      """

    def __init__(self, data_set_path, data_points, radius, batch_size=32, shuffle=True):
        self.data_set_path = data_set_path
        self.data_points = data_points
        self.radius = radius
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Get encoding for atomic numbers and amino acids:
        self.data_encoder, self.label_encoder = encode_data()

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data_points) / self.batch_size))

    def __getitem__(self, index):
        dims = (
            self.radius * 2 + 1,
            self.radius * 2 + 1,
            self.radius * 2 + 1,
            len(self.data_encoder.categories_[0]),
        )
        X = np.empty((self.batch_size, *dims), dtype=np.uint8)
        y = np.empty(self.batch_size, dtype="|S3")
        data_point_batch = self.data_points[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        data = []
        labels = []
        with h5py.File(str(self.data_set_path), "r") as dataset:
            for i, (pdb, indices, label, _) in enumerate(data_point_batch):
                data = np.pad(dataset[pdb]["data"], self.radius, mode="constant")
                shape = data.shape
                indices = [
                    indices[0] + self.radius,
                    indices[1] + self.radius,
                    indices[2] + self.radius,
                ]
                region = data[
                    indices[0] - self.radius : indices[0] + self.radius + 1,
                    indices[1] - self.radius : indices[1] + self.radius + 1,
                    indices[2] - self.radius : indices[2] + self.radius + 1,
                ]
                shape = region.shape
                X[i,] = self.data_encoder.transform(
                    region.flatten().reshape(-1, 1)
                ).reshape(*shape, -1)
                y[i,] = label
        encoded_y = self.label_encoder.transform(y.reshape(-1, 1))
        return X, encoded_y

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.data_points)


class ContigDiscretizedProteinsSequence(keras.utils.Sequence):
    """
      Keras Sequence object containing contigs of protein frames.

      Each of the protein frames is voxelised using the label encoder. The
      alpha carbons are centered within these frames.

      Extra padding is added to the frame to obtain a frame of specified
      radius.

      Units value specify how many protein frames are inserted into a contig

      Attributes
      ----------
      data_set_path : Path Object or str
          Path to structural data

      data_points : array of tuples
          [ (PDB Code, [Ca coordinates in n units], residue identity of Ca,
          atom encoder) ]

          eg.
          [('5er6.pdb1',
              [(55, 38, 59),
               (58, 43, 62),
               (63, 43, 58),
               (60, 44, 52),
               (57, 48, 55),
               (62, 51, 58),
               (65, 51, 52),
               (60, 54, 49),
               (59, 58, 53),
               (65, 60, 54),
               (66, 60, 48)],
              'TRP',
              {0, 6, 7, 8}),
          ...]

      units : int
          Number of frames units in one contig

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

      batch_size: int
          Number of data_points considered at once

      shuffle: bool
          Shuffling of the order of data_points
      """

    def __init__(
        self, data_set_path, data_points, units, radius, batch_size=32, shuffle=True,
    ):
        self.data_set_path = data_set_path
        self.data_points = data_points
        self.units = units
        self.radius = radius
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Get encoding for atomic numbers:
        self.data_encoder, self.label_encoder = encode_data()

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data_points) / self.batch_size))

    def __getitem__(self, index):

        dims = (
            self.radius * 2 + 1,
            self.radius * 2 + 1,
            self.radius * 2 + 1,
            len(self.data_encoder.categories_[0]),
        )

        X = np.empty((self.batch_size, self.units, *dims), dtype=np.uint8)
        y = np.empty(self.batch_size, dtype="|S3")

        data_point_batch = self.data_points[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        data = []
        labels = []
        with h5py.File(str(self.data_set_path), "r") as dataset:
            for i, (pdb, indices_list, label, _) in enumerate(data_point_batch):
                # Add padding
                data = np.pad(dataset[pdb]["data"], self.radius, mode="constant")

                shape = data.shape
                for j, indices in enumerate(indices_list):

                    if indices is None:
                        X[i, j] = np.zeros(dims, dtype=np.uint8)

                    else:
                        # Center alpha carbon in frame center
                        region = data[
                            indices[0] - self.radius : indices[0] + self.radius + 1,
                            indices[1] - self.radius : indices[1] + self.radius + 1,
                            indices[2] - self.radius : indices[2] + self.radius + 1,
                        ]

                        shape = region.shape

                        X[i, j] = self.data_encoder.transform(
                            region.flatten().reshape(-1, 1)
                        ).reshape(*shape, -1)

                y[i] = label

        encoded_y = self.label_encoder.transform(y.reshape(-1, 1))

        return X, encoded_y

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.data_points)


def annotate_data_with_frame_prediction(data_points, radius, data_set_path, model_path):
    """"
    data_points: array of tuples

          [ (PDB Code, [Ca coordinates in n units], residue identity of Ca,
          atom encoder) ]

          eg.
          [('5er6.pdb1',
              [(55, 38, 59),
               (58, 43, 62),
               (63, 43, 58),
               (60, 44, 52),
               (57, 48, 55),
               (62, 51, 58),
               (65, 51, 52),
               (60, 54, 49),
               (59, 58, 53),
               (65, 60, 54),
               (66, 60, 48)],
              'TRP',
              {0, 6, 7, 8}),
          ...]


    radius: int
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

    data_set_path: Path or str
        Path to structural data

    model_path: Path or str
        Path to the model to pre-fill predictions


    Returns
    -------
    converted_data_points: array of tuples

        (PDB Code, units of amino acid probabilities, amino acid identity)

        [('5er6.pdb1',

          array([[0.04900201, 0.05389743, 0.07173565, 0.0567386 , 0.0268912 ,
                  0.06304525, 0.07155736, 0.02619088, 0.0638176 , 0.02624071,
                  0.04347818, 0.05504276, 0.04269378, 0.0374709 , 0.03032501,
                  0.10125927, 0.05716411, 0.05376817, 0.04731638, 0.0223647 ],
                 [0.1126359 , 0.13956866, 0.10388297, 0.02143906, 0.04641615,
                  0.05595287, 0.02393122, 0.02494205, 0.04219558, 0.02188579,
                  0.02607852, 0.05184305, 0.04044017, 0.02445424, 0.02877329,
                  0.11255109, 0.03344607, 0.03635255, 0.02763311, 0.02557763],
                 [0.07159909, 0.03839176, 0.03355072, 0.04304969, 0.09471414,
                  0.04893323, 0.0455103 , 0.0245069 , 0.04385265, 0.03113862,
                  0.03987258, 0.03622015, 0.04674242, 0.07340036, 0.02011707,
                  0.05934293, 0.02794214, 0.11794958, 0.0686977 , 0.034468  ],
                 [0.04188311, 0.04942966, 0.04711344, 0.06561235, 0.02426023,
                  0.06758102, 0.10206639, 0.02020826, 0.03914877, 0.04511404,
                  0.06612615, 0.05174561, 0.03752192, 0.04650718, 0.06710701,
                  0.04523845, 0.03911253, 0.05940424, 0.05059638, 0.0342233 ],
                 [0.07322103, 0.02532865, 0.03024493, 0.0308362 , 0.14460151,
                  0.03894044, 0.03600531, 0.02851084, 0.03478802, 0.05524605,
                  0.07505813, 0.02646575, 0.06339504, 0.06145038, 0.02035042,
                  0.04377752, 0.03951412, 0.04789652, 0.06190815, 0.06246092],
                 [0.06851989, 0.05284749, 0.06413279, 0.05878077, 0.05487762,
                  0.06252896, 0.06840792, 0.03276298, 0.0441237 , 0.02348907,
                  0.03585042, 0.05077354, 0.05522553, 0.04669135, 0.02295987,
                  0.09432437, 0.04482105, 0.04962397, 0.04501988, 0.02423875],
                 [0.06043382, 0.07376552, 0.06050602, 0.06138247, 0.02205661,
                  0.09372416, 0.11755891, 0.02180074, 0.03821517, 0.02616759,
                  0.04363715, 0.07496871, 0.0484312 , 0.032888  , 0.01791682,
                  0.07307338, 0.04694738, 0.03067271, 0.0302284 , 0.02562528],
                 [0.0548516 , 0.05558271, 0.04647261, 0.0461969 , 0.03550095,
                  0.07030181, 0.06872175, 0.02162899, 0.03949278, 0.0411206 ,
                  0.09642904, 0.05458071, 0.07616292, 0.05189585, 0.01877855,
                  0.04785829, 0.03374705, 0.05408014, 0.05911661, 0.02748019],
                 [0.10540234, 0.02058502, 0.02888341, 0.01852538, 0.16288903,
                  0.02711768, 0.02351689, 0.01993977, 0.02955796, 0.0653744 ,
                  0.04087251, 0.02038115, 0.0467665 , 0.03053688, 0.01694887,
                  0.04127438, 0.07153568, 0.02404181, 0.0366751 , 0.16917522],
                 [0.08624613, 0.04818901, 0.04529418, 0.02824474, 0.07543088,
                  0.04649773, 0.04309331, 0.02331424, 0.0397668 , 0.03739208,
                  0.07465584, 0.03703914, 0.07642883, 0.05442159, 0.01945952,
                  0.06646051, 0.0458856 , 0.04720572, 0.05105588, 0.05391829],
                 [0.07085727, 0.06865241, 0.07675168, 0.06099009, 0.02621241,
                  0.07984767, 0.08752365, 0.04338133, 0.03592592, 0.02247998,
                  0.04262917, 0.05311032, 0.05794421, 0.03603866, 0.02050281,
                  0.08654983, 0.03484168, 0.04181154, 0.0328845 , 0.02106488]],
                dtype=float32),

        array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 1., 0., 0.]])),
        ..]

    """
    # Get encoding for atomic numbers and amino acids:
    data_encoder, label_encoder = encode_data()

    dims = (
        radius * 2 + 1,
        radius * 2 + 1,
        radius * 2 + 1,
        len(data_encoder.categories_[0]),
    )
    converted_data_points = []

    with h5py.File(str(data_set_path), "r") as dataset:
        for pdb, indices_list, label, _ in data_points:
            X = np.empty((len(indices_list), *dims), dtype=np.uint8)
            data = np.pad(dataset[pdb]["data"], radius, mode="constant")
            shape = data.shape
            for i, indices in enumerate(indices_list):
                if indices is None:
                    X[i] = np.zeros(dims, dtype=np.uint8)
                else:
                    region = data[
                        indices[0] - radius : indices[0] + radius + 1,
                        indices[1] - radius : indices[1] + radius + 1,
                        indices[2] - radius : indices[2] + radius + 1,
                    ]
                    # Check that the central atom is a CA
                    assert region[radius][radius][radius] == 6
                    shape = region.shape
                    X[i] = data_encoder.transform(
                        region.flatten().reshape(-1, 1)
                    ).reshape(*shape, -1)

            # Load model to prefill predictions
            model = load_model(model_path)
            predictions = model.predict(X)

            for i, indices in enumerate(indices_list):
                if indices is None:
                    predictions[i] = np.zeros((20,))

            encoded_label = label_encoder.transform(np.array([label]).reshape(-1, 1))
            converted_data_points.append((pdb, predictions, encoded_label))

    return converted_data_points


def make_data_points(
    data_set_path,
    pdb_codes,
    radius,
    uncommon_res_conversion=UNCOMMON_RES_CONVERSION,
    shuffle=True,
):
    """
    Creates frames of structures with specified radius and centers the Ca in the middle of the frame.

    Parameters
    ----------
    data_set_path : Path
        Path to h5 dataset of structures
    pdb_codes : List of str
        List of PDB codes to be framed.
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
    uncommon_res_conversion : Bool
        Bool of whether the program will attempt to convert the uncommon
        residues to a common one.
    shuffle: bool
        Shuffling of the order of data_points

    Returns
    -------
    data_points : array of tuples
        [(PDB Code, [Ca coordinates], residue identity of Ca, atom encoder)]
        eg.
        [
        '5c7h.pdb1',
        (55, 61, 41),
        'LYS',
        array([0, 6, 7, 8])
        ...]

    """
    data_points = []
    standard_aas = standard_amino_acids.values()
    with h5py.File(data_set_path, "r") as data_set:
        for pdb in pdb_codes:

            group = data_set[pdb]

            #  group['indices'] store the coords of Ca atoms
            assert len(group["indices"]) == len(
                group["labels"]
            ), "Should have same number of indices and labels"

            #  Add padding
            data = np.pad(data_set[pdb]["data"], radius, mode="constant")
            for i, l in zip(group["indices"], group["labels"]):
                # Decode from bytes to unicode
                decoded_l = l.decode()

                # Check if center aa is standard:
                if decoded_l not in standard_aas and uncommon_res_conversion:
                    if decoded_l in UNCOMMON_RESIDUE_DICT.keys():
                        print(
                            f"ATTENTION: We are converting {decoded_l} to "
                            f"{UNCOMMON_RESIDUE_DICT[decoded_l]}. "
                        )
                        decoded_l = UNCOMMON_RESIDUE_DICT[decoded_l]
                    else:
                        assert decoded_l in standard_aas, (
                            f"Expected standard residue values, attempted "
                            f"conversion, but got {decoded_l}."
                        )
                elif not uncommon_res_conversion:
                    assert decoded_l in standard_aas, (
                        f"Expected standard residue values, but got "
                        f"{decoded_l}, uncommon residue conversion is off."
                    )

                # Centers the Ca carbon to create frame with Ca at center
                padded_indices = (i[0] + radius, i[1] + radius, i[2] + radius)
                region_data = data[
                    padded_indices[0] - radius : padded_indices[0] + radius + 1,
                    padded_indices[1] - radius : padded_indices[1] + radius + 1,
                    padded_indices[2] - radius : padded_indices[2] + radius + 1,
                ]
                #  i is the coordinates of Ca (immutable as tuple)
                #  Unique classes in voxels
                data_points.append((pdb, tuple(i), decoded_l, np.unique(region_data)))

        if shuffle:
            random.shuffle(data_points)

    return data_points
