import os
import json
import numpy as np
import matplotlib.pyplot as plt
import six
import itertools

def list_folders(path):
    assert os.path.exists(path), 'The provided path does not exist'
    for folder in os.listdir(path):
        d_path = os.path.join(path, folder)
        if os.path.isdir(d_path):
            yield folder, d_path


def list_files(path, ftype='.json'):
    assert os.path.exists(path), 'The provided path does not exist'
    for file in os.listdir(path):
        if file.endswith(ftype):
            yield os.path.join(path, file)


def parse_data(data_path):
    assert os.path.exists(data_path), 'The provided path does not exist'
    for f_name, d_path in list_folders(data_path):
        for file in list_files(d_path):
            with open(file) as data_file:
                yield json.load(data_file), f_name


def filter_by_pkey(_dict, pkey):
    """
        Filter dictionary by an incomplete key
    """
    return [v for k, v in _dict.iteritems() if pkey in k]


def filter_by_key_list(_dict, klist):
    return {k: v[0] for k, v in _dict.iteritems() if k in klist}


def filter_data(data, key, feat_keys):
    data_f = filter_by_pkey(data, key)
    return filter_by_key_list(data_f, feat_keys)


def render_mpl_table(data, col_width=1.0, row_height=0.3, font_size=11,
         header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
         bbox=[0, 0, 1, 1], header_columns=0,
         ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def stratified_split(y, train_ratio):

    def split_class(y, label, train_ratio):
        indices = np.flatnonzero(y == label)
        n_train = int(indices.size*train_ratio)
        train_index = indices[:n_train]
        test_index = indices[n_train:]
        return (train_index, test_index)

    idx = [split_class(y, label, train_ratio) for label in np.unique(y)]
    train_index = np.concatenate([train for train, _ in idx])
    test_index = np.concatenate([test for _, test in idx])
    return train_index, test_index