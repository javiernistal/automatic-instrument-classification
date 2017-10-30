import argparse
import pickle
import logging
import numpy as np

from datetime import datetime
from os.path import join

from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report

from instrument_classification.tools import *
from instrument_classification.headers import raw_feat_keys
from instrument_classification import EstimatorSelection
from instrument_classification.gs_params import models, dim_red, dparams, mparams


def read_filt_data(path):
    """Reads and filters the dataset in path
    Args:
        path: path to the dataset

    Returns:
        a list containing feature data instances and a list of targets
    """
    dataset = []
    targets = []
    for data, instr_cl in parse_data(path):
        for smpl_feats in filter_by_pkey(data, 'sample_'):
            targets.append(instr_cl)
            dataset.append(
                np.hstack(filter_by_key_list(smpl_feats, raw_feat_keys).values())
            )
    return dataset, targets

def scale_norm(data): 
            return MinMaxScaler().fit_transform(StandardScaler().fit_transform(data))

def save_model_metrics(estimator, name, scores, cm, classes, path):
    
    
    # Save score summary
    render_mpl_table(scores, header_columns=0, col_width=2.0)
    plt.savefig(join(path, 'summary_score.png'))
    
    # Save estimator
    with open(join(path, name + '.p'), 'wb') as pkl:
        pickle.dump(estimator, pkl)

    # Save confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, classes) 
    plt.savefig(join(path, name + '.png'))


def main(args):
    dtime = datetime.now().strftime('%H_%M_%d_%m_%Y')
    # Create folder for experiment results
    rpath = 'results'
    
    fpath = join(rpath, args.name + '_' + dtime)
    create_folder(fpath)

    # Read data and targets. 
    X, target = read_filt_data(args.path)
    
    cl   = np.unique(target)
    n_cl = len(cl)

    # Map instrument class target ('piano', ..., 'saxo') to a numeric 
    # label (0, ..., 8)
    cl_label_dict = dict(zip(cl, range(0, n_cl)))
    label_cl_dict = {v: k for k, v in cl_label_dict.iteritems()}
    y = [cl_label_dict[e] for e in target]

    # Create an estimator selector with the parameters set in grid_search.py
    estimator_sel = EstimatorSelection(
        models, mparams, dim_red, dparams, log_name=join(fpath, args.name + '.log')
    )

    # Scale and normalize data [0,1]
    Xn = scale_norm(X)

    # Train and evaluate estimators
    estimator_sel.fit(Xn, y, cv=5)

    # Generate estimator selection score results and get best one
    score_results = estimator_sel.score_summary()
    best_estimator = estimator_sel.get_best_estimator(score_results)

    # Evaluate best estimator
    pred = cross_val_predict(best_estimator, Xn, y, cv=5)
    cm, classes = confusion_matrix([label_cl_dict[e] for e in y], [label_cl_dict[p] for p in pred]), np.unique(target)
    cl_report = classification_report([label_cl_dict[e] for e in y], [label_cl_dict[p] for p in pred])

    # Save performance metrics
    logging.info('Performance metrics:\n\n %s', cl_report)
    logging.info('\nCONFUSION MATRIX\n: %s', cm)
    save_model_metrics(best_estimator, 'confusion_matrix_' + dtime, score_results, cm, classes, fpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimator selection for Instrument classification.')
    parser.add_argument(
        '-p', dest='path', type=str, required=True,
            help='path to the instrument dataset'
    )
    parser.add_argument(
        '-n', dest='name', type=str, required=True,
            help='name of the experiment output folder'
    )
    main(parser.parse_args())
