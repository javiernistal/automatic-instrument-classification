from tools import *
from headers import raw_feat_keys
import numpy as np
import argparse
import pickle
import time

from model_estimator import EstimatorSelection
from grid_search import models, dim_red, dparams, mparams, preprocess


def read_filt_data(path):
    dataset = []
    targets = []
    for data, instr_cl in parse_data(path):
        for smpl_feats in filter_by_pkey(data, 'sample_'):
            targets.append(instr_cl)
            dataset.append(
                np.hstack(filter_by_key_list(smpl_feats, raw_feat_keys).values())
            )
    return dataset, targets


def main(args):

    X, target = read_filt_data(args.path)
    n_cl = len(np.unique(target))
    cl_label_dict = dict(zip(np.unique(target), range(0, n_cl)))
    y  = [cl_label_dict[e] for e in target]

    estimator = EstimatorSelection(models, mparams, dim_red, dparams, preprocess)

    estimator.fit(X, y, cv=5)
    score_results = estimator.score_summary()
    print score_results

    _time = time.ctime()[4:-5]
    _time = _time.replace(':', '_')
    score_results.to_csv('score_'+_time+'.csv')

    render_mpl_table(estimator.score_summary(), header_columns=0, col_width=2.0)
    plt.savefig('file'+_time+'.png')
    pickle.dumps(estimator.build_best_model(score_results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '-p', dest='path', type=str,
            help='an integer for the accumulator'
    )
    # parser.add_argument(
    #     '-p', dest='out_path', type=str,
    #         help='an integer for the accumulator'
    # )
    main(parser.parse_args())
