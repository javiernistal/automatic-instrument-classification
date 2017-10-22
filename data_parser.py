import os

from os import listdir
from os.path import isfile, join, isdir

import sklearn as sk
import pandas as pd
import numpy as np
import json


from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
def main():
    feat_keys = [
        'mfcc',
        'gfcc',
        'energy',
        'flux',
        'spectralComplexity',
        'centroid',
        'flatness',
        'zeroCrossingRate',
        'lpc'
    ]

    feat_keys_all = [
        'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13',
        'gfcc_1', 'gfcc_2', 'gfcc_3', 'gfcc_4', 'gfcc_5', 'gfcc_6', 'gfcc_7', 'gfcc_8', 'gfcc_9', 'gfcc_10', 'gfcc_11', 'gfcc_12', 'gfcc_13',
        'energy',
        'flux',
        'spectralComplexity',
        'centroid',
        'flatness',
        'zeroCrossingRate',
        'lpc_1', 'lpc_2', 'lpc_3', 'lpc_4', 'lpc_5', 'lpc_6', 'lpc_7', 'lpc_8', 'lpc_9', 'lpc_10', 'lpc_11', 
        'target'
    ]

    data_path = '../music_group_ml_test/music_group_ml_test_data/'
    def read_jsons(data_path):
        
        for d in os.listdir(data_path):
            d_path = join(data_path, d)
            if isdir(d_path):
                target = d
                for f in os.listdir(d_path):
                    f_path = join(d_path, f)
                    if f.endswith('.json'):
                        with open(f_path) as data_file:    
                            data = json.load(data_file)
                        yield data, target
                        
    dataset = []
    for data_file, target in read_jsons(data_path):

        for key in data_file.keys():
            if key.startswith('sample_'):
                item_dict = {k: v[0] for k, v in data_file[key].iteritems() if k in feat_keys}
                dataset.append(list(np.hstack(item_dict.values())) + [target])

    from sklearn.preprocessing import StandardScaler

    data = pd.DataFrame(columns=feat_keys_all, data=dataset)
    data.index = data.target

    X = data.iloc[:, :-1]
    y = data.target

    class_label_dict = dict(zip(np.unique(y), range(0, len(np.unique(y)))))
    y_t = [class_label_dict[e] for e in y]


    scaler = StandardScaler()
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)


    # X = SelectKBest(chi2, k=40).fit_transform(X, y)



    del data['target']

    import pandas as pd
    from sklearn.grid_search import GridSearchCV
    from numpy import mean, std
    from sklearn.decomposition import PCA, NMF

    class EstimatorSelectionHelper:
        def __init__(self, models, params, dim_red, params2):
            if not set(models.keys()).issubset(set(params.keys())):
                missing_params = list(set(models.keys()) - set(params.keys()))
                raise ValueError("Some estimators are missing parameters: %s" % missing_params)
            self.models = models
            self.params = params
            self.keys = models.keys()
            self.grid_searches = {}
            
            self.dim_red = dim_red
            self.keys2 = dim_red.keys()
            self.params2 = params2
        
        def fit(self, X, y, cv=5, n_jobs=1, verbose=1, scoring=None, refit=False):
            for key2 in self.keys2:
                self.grid_searches[key2] = {}
                for key in self.keys:
                    print("Running GridSearchCV for %s." % key)
                    model = self.models[key]
                    
                    
                    
                    pipe = Pipeline(steps=[('dim_red', self.dim_red[key2]), ('classifier', model)])
                    
                    
                    params = None
                    params = dict(self.params[key].items() + self.params2[key2].items())
                    
                    gs = GridSearchCV(pipe, params, cv=cv, n_jobs=n_jobs, 
                                      verbose=verbose, scoring=scoring, refit=refit)
                    gs.fit(X,y)
                    

                    self.grid_searches[key2][key] = gs

        def score_summary(self, sort_by='mean_score'):
            def row(key, key2, scores, params):
                d = {
                     'estimator' : key,
                     'dim_red'   : key2,
                     'min_score' : round(min(scores) , 3),
                     'max_score' : round(max(scores) , 3),
                     'mean_score': round(mean(scores), 3),
                     'std_score' : round(std(scores) , 3),
                }
                return pd.Series(dict(params.items() + d.items()))
            rows = []
            for k2 in self.keys2:
                for k in self.keys:
                    for gsc in self.grid_searches[k2][k].grid_scores_:
                        model_res = row(
                            k, k2, gsc.cv_validation_scores, gsc.parameters
                        )
                        rows.append(model_res)
            
            
            df = (pd.concat(rows, axis=1).T)
    #         df = pd.DataFrame(rows)
            df = df.sort_values([sort_by], ascending=False)
            
            columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
            columns = columns + [c for c in df.columns if c not in columns]
            
            return df[columns]

    from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier, 
                                  AdaBoostClassifier, GradientBoostingClassifier)
    from sklearn.svm import SVC

    models1 = { 
        'ExtraTreesClassifier': ExtraTreesClassifier(),
         'RandomForestClassifier': RandomForestClassifier(),
        # 'AdaBoostClassifier': AdaBoostClassifier(),
        # 'GradientBoostingClassifier': GradientBoostingClassifier(),
        'SVC': SVC()
    }

    params1 = { 
        'ExtraTreesClassifier': { 'classifier__n_estimators': [16, 32] },
        'RandomForestClassifier': { 'classifier__n_estimators': [16, 32] },
        # 'AdaBoostClassifier':  { 'classifier__n_estimators': [16, 32] },
        # 'GradientBoostingClassifier': { 'classifier__n_estimators': [16, 32], 'classifier__learning_rate': [0.5, 0.1] },
        'SVC': {'classifier__kernel': ['linear', 'rbf'], 'classifier__C': [10000, 100000], 'classifier__gamma':  [0.001, 0.01]}
    #         {'kernel': ['rbf'], 'C': [1, 100, 10000, 100000], 'gamma': [0.001, 0.0001]},
        
    }
    dim_red = {
        'pca': PCA(),
        'nmf': NMF(),
        'kbest': SelectKBest(chi2)
    }
    params2 = {
        'pca': {'dim_red__n_components': [30, 40]},
        'nmf': {'dim_red__n_components': [30, 40]} ,
        'kbest': {'dim_red__k': [30, 40]}
                      }

    helper1 = EstimatorSelectionHelper(models1, params1, dim_red, params2)
    helper1.fit(X, y_t, n_jobs=2)
    print helper1.score_summary(sort_by='min_score')



    import pandas
    import six
    import seaborn as sns
    helper1.score_summary().to_csv('test3.csv')
    from pandas.tools.plotting import table
    # ax = plt.subplot(111, frame_on=False) # no visible frame
    # ax.xaxis.set_visible(False)  # hide the x axis
    # ax.yaxis.set_visible(False)  # hide the y axis

    # table(ax, helper1.score_summary())  # where df is your data frame

    # plt.savefig('mytable.png')

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

    render_mpl_table(helper1.score_summary(), header_columns=0, col_width=2.0)
    plt.savefig('file2.png')


if __name__ == '__name__':
    main()
