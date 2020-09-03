import numpy as np

from sklearn.metrics import confusion_matrix


def make_frame_confusion_matrix(model, data, batches):
    y_true = []
    y_predicted = []

    for i in range(batches):
        # Extract data:
        data_point, labels = data[i]

        y_true.extend(np.argmax(labels, axis=1))
        # Predict label
        y_predicted.extend(np.argmax(model.predict(data_point), axis=1))

    cm = confusion_matrix(y_true, y_predicted)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    return cm


def make_contig_confusion_matrix(model, data, count):
    y_true = []
    y_predicted = []

    # Extract data:
    data, labels = data

    y_true.extend(np.argmax(labels[:count], axis=1))
    # Predict Label
    y_predicted.extend(np.argmax(model.predict(data[:count]), axis=1))
    # Compare predicted vs true label
    cm = confusion_matrix(y_true, y_predicted)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    return cm
