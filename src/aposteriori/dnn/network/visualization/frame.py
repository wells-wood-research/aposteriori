from aposteriori.dnn.config import (
    ACTIVATION_ALPHA,
    ATOM_COLORS,
    COLOR_MAP,
    FIG_SIZE,
    FRAME_CONV_MODEL,
    LOCAL_COLOR_MAP,
    PLOT_DIR,
)
from aposteriori.dnn.data_processing.encoder import encode_data

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.interpolation import zoom
from tensorflow.keras.models import Model


def _normalize_array(data_array):
    """
    Normalizes values of array between 0 - 1.

    Parameters
    ----------
    data_array : numpy.ndarray
        Array of floats with various ranges (may also be negative)
    Returns
    -------
    normalized array : numpy.ndarray
        Normalized array with range of values 0 - 1
    """
    return (data_array - data_array.min()) / (data_array.max() - data_array.min())


def _get_frame_atom_coordinates(frame_array, data_encoder):
    """
    Returns the coordinates of atoms to be plotted in the frame.

    Parameters
    ----------
    frame_array : numpy.ndarray
        Array (frame_radius_x, frame_radius_y, frame_radius_z,
        atomic_numbers) where xyz coordinates of each atoms are stored
    data_encoder : sklearn.preprocessing.OneHotEncoder
        Atom encoder used to extract xyz coordinates of specific atom in voxel

    Returns
    -------
    atom_coords : dict
        Dictionary of atoms to coords {Atomic Number : [ [X], [Y], [Z] ]}

    """
    atom_coords = {}

    # Represent atoms: (skips 0 - empty space)
    for i in range(1, len(np.unique(data_encoder.categories_[0]))):
        atom_slice = frame_array[:, :, :, i]
        atom_x = []
        atom_y = []
        atom_z = []

        for ix, yzs in enumerate(atom_slice):
            for iy, zs in enumerate(yzs):
                for iz, a in enumerate(zs):
                    if a == 0:
                        continue
                    else:
                        atom_x.append(ix)
                        atom_y.append(iy)
                        atom_z.append(iz)

        atom_coords[data_encoder.categories_[0][i]] = [atom_x, atom_y, atom_z]

    return atom_coords


def _visualize_frame(
    activation_array,
    frame_array,
    calculated_residue_probability,
    real_residue,
    frame_index,
    local_color_map=LOCAL_COLOR_MAP,
):
    """
    Visualizes the frame and the activation layer.

    Parameters
    ----------
    activation_array : numpy.ndarray
        Result of frame after activation layer (nx, ny, nz, w) where w is the
        intensity of activation and n represents spatial coordinates
    frame_array : numpy.ndarray
        Frame array (frame_radius_x, frame_radius_y, frame_radius_z,
        atomic_numbers) where xyz coordinates of each atoms are stored
    calculated_residue_probability : float
        Probability output from model for a given frame
    real_residue : str
        Identity of the real residue in frame
    frame_index : int
        Index of frame to be visualized
    local_color_map : bool
        Whether the color map of the activation considers local (True)
        maximum or overall maxium (False)

    """
    # Get data encoder:
    data_encoder, label_encoder = encode_data()

    # Get atom coordinates in frame:
    atom_coords = _get_frame_atom_coordinates(frame_array, data_encoder)

    # Normalize the array from 0 to 1:
    normalized_array = _normalize_array(activation_array)
    # Zoom the activation layer to be the same shape of the input frame:
    zoom_factor = frame_array.shape[0] / activation_array.shape[0]

    # The intensity (x,y,z, intensity) doesn't need to be altered, therefore
    # its zoom value is 1
    resized_array = zoom(normalized_array, (zoom_factor, zoom_factor, zoom_factor, 1))

    # Plot each amino acid:
    for i in range(0, len(label_encoder.categories_[0])):
        frame_slice = resized_array[:, :, :, i]

        # Extract activation Coordinates:
        x = []
        y = []
        z = []
        w = []
        for ix, yzs in enumerate(frame_slice):
            for iy, zs in enumerate(yzs):
                for iz, a in enumerate(zs):
                    x.append(ix)
                    y.append(iy)
                    z.append(iz)
                    w.append(a)

        # Create figure
        fig = plt.figure(figsize=FIG_SIZE)
        ax = fig.add_subplot(121, projection="3d")

        # Label the real residue present in the position
        if label_encoder.categories_[0][i] == real_residue:
            is_correct_residue = " Real"
        else:
            is_correct_residue = ""

        # Plot Settings:
        ax.set_title(
            label_encoder.categories_[0][i]
            + " "
            + str(round(calculated_residue_probability[i] * 100, 2))
            + is_correct_residue,
            fontsize=12,
        )
        ax.set_ylim([0, frame_array.shape[0]])
        ax.set_xlim([0, frame_array.shape[0]])
        ax.set_zlim([0, frame_array.shape[0]])

        # Add axes labels:
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis", labelpad=15)

        # Colorbar:
        # Normalize activation colors to be min and max of the local activation
        if local_color_map:
            norm = colors.Normalize(vmin=min(w), vmax=max(w))
        else:
            norm = colors.Normalize(vmin=min(resized_array), vmax=max(resized_array))

        sm = plt.cm.ScalarMappable(cmap=COLOR_MAP, norm=norm)
        fig.colorbar(sm).set_label("Attention Level (Range 0 - 1)")

        # Plot Activation + Set color to intensity:
        ax.scatter(
            x,
            y,
            z,
            zdir="z",
            alpha=ACTIVATION_ALPHA,
            color=COLOR_MAP(norm(w)),
            depthshade=False,
        )

        # Plot Atoms:
        for k in atom_coords.keys():
            x, y, z = atom_coords[k]

            ax.scatter(x, y, z, zdir="z", color=ATOM_COLORS[k], depthshade=False)

        # Save
        plt.savefig(
            PLOT_DIR
            / (label_encoder.categories_[0][i] + f"_frame_{frame_index}" + ".png"),
            bbox_inches="tight",
            pad_inches=0.3,
            quality=95,
        )


def visualize_model_layer(
    layer_depth, frame_set, frame_index, frame_model_path=FRAME_CONV_MODEL
):
    """
    Visualizes specific layers (usually activation) in a frame.

    Parameters
    ----------
    layer_depth : int
        Spefiying the layer to be visualized
    frame_set : FrameDiscretizedProteinsSequence
        Class set of frames of voxelised proteins
    frame_index : int or list of int
        Index (or indices) of frames to be visualized
    frame_model_path : Path or str to frame model


    """
    frame_model = tf.keras.models.load_model(frame_model_path)
    print(
        f"Visualizing layer {layer_depth}, "
        f"{frame_model.layers[layer_depth].name}, with output shape, "
        f"{frame_model.layers[layer_depth].output.shape}"
    )

    # Create models (both activation and final)
    activation_model = Model(
        inputs=frame_model.inputs, outputs=frame_model.layers[layer_depth].output
    )

    # Calculate predictions:
    activation_prediction = activation_model.predict_generator(frame_set)
    final_prediction = frame_model.predict_generator(frame_set)

    # Visualize index if integer
    if isinstance(frame_index, int):
        _visualize_frame(
            activation_prediction[frame_index],
            frame_set[frame_index][0][0],
            final_prediction[frame_index],
            frame_set.data_points[frame_index][2],
            frame_index,
        )

    # Visualize multiple indeces if list of integers
    elif isinstance(frame_index, list) and isinstance(frame_index[0], int):
        for i in frame_index:
            _visualize_frame(
                activation_prediction[i],
                frame_set[i][0][0],
                final_prediction[i],
                frame_set.data_points[i][2],
                i,
            )

    else:
        print(
            f"Unsupported Type {type(frame_index)}, supported types are "
            f"integers or list of integers"
        )


def visualize_frame_of_voxels(frame_voxels):
    """
    Produces a plot of the voxels in a frame. Used mostly for debugging
    purposes.

    :param frame_voxels: Array of floats
        The frame of voxels. The shape of the frame is (y, x, z, atoms)
    """

    for atom_idx in range(len(frame_voxels[0, 0, 0, :])):
        x = []
        y = []
        z = []
        w = []
        # Extract x y z and density values:
        for ix, yzs in enumerate(frame_voxels[:, :, :, atom_idx]):
            for iy, zs in enumerate(yzs):
                for iz, a in enumerate(zs):
                    x.append(ix)
                    y.append(iy)
                    z.append(iz)
                    w.append(a)
        norm = colors.Normalize(vmin=min(w), vmax=max(w))
        fig = plt.figure()
        ax = fig.add_subplot(121, projection="3d")
        ax.scatter(
            x, y, z, zdir="z", alpha=0.4, color=COLOR_MAP(norm(w)), depthshade=False,
        )
        plt.show()
