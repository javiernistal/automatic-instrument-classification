from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from numpy import mean, std
import pandas as pd
from grid_search import norm
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class EstimatorSelection(object):
    def __init__(self, models, mparams, dim_red, dparams, preprocess):
        if not set(models.keys()).issubset(set(mparams.keys())):
            missing_params = list(set(models.keys()) - set(mparams.keys()))
            raise ValueError(
                "Some estimators are missing parameters: %s" % missing_params
            )

        self.models     = models
        self.mparams    = mparams
        self.mkeys      = models.keys()
        self.dim_r      = dim_red
        self.dkeys      = dim_red.keys()
        self.dparams    = dparams
        self.grid_s     = {}
        self.preprocess = preprocess

    def fit(self, X, y, cv=5, n_jobs=1, verbose=1, scoring=None, refit=False):
        def normalization(data): 
            return MinMaxScaler().fit_transform(StandardScaler().fit_transform(data))
        X = normalization(X)
        for dkey in self.dkeys:
            self.grid_s[dkey] = {}
            for mkey in self.mkeys:
                print("Running GridSearchCV for %s." % mkey)
                
                model  = self.models[mkey]
                pipe   = Pipeline(steps=[('dim_red', self.dim_r[dkey]), ('classifier', model)])
                params = dict(self.mparams[mkey].items() + self.dparams[dkey].items())
                gs     = GridSearchCV(pipe, params, cv=cv, n_jobs=n_jobs, 
                                  verbose=verbose, scoring=scoring, refit=refit)
                gs.fit(X,y)
                self.grid_s[dkey][mkey] = gs

    def score_summary(self, sort_by='mean_score', n=10):
        def row(mkey, dkey, scores, params):
            d = {
                 'model'     : mkey,
                 'dim_red'   : dkey,
                 'min_score' : round(min(scores) , 3),
                 'max_score' : round(max(scores) , 3),
                 'mean_score': round(mean(scores), 3),
                 'std_score' : round(std(scores) , 3),
            }
            return pd.Series(dict(params.items() + d.items()))
        rows = []
        for dk in self.dkeys:
            for mk in self.mkeys:
                for gsc in self.grid_s[dk][mk].grid_scores_:
                    model_res = row(
                        mk, dk, gsc.cv_validation_scores, gsc.parameters
                    )
                    rows.append(model_res)
        df = (pd.concat(rows, axis=1).T)
        df = df.sort_values([sort_by], ascending=False).head(n)
        
        columns = ['model', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]
        
        return df[columns]


    def build_best_model(self, score_sum):
        mkey = score_sum['model'][0]
        dkey = score_sum['dim_red'][0]
        print mkey, dkey
        return self.grid_s[dkey][mkey].estimator.set_params(**self.grid_s[dkey][mkey].best_params_)