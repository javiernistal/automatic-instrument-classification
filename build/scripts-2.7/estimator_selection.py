import logging
import pandas as pd

from os.path import exists, dirname
from numpy import mean, std

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline


class EstimatorSelection(object):
    def __init__(self, models, mparams, dim_red, dparams, log_name):
        """Initializes EstimatorSelection class
        
        Args:
            models:   dict containing the models to be compared
            mparams:  dict containing the parameters on which to run the
                      grid search for each model
            dim_red:  dict containing the dimensionality reduction methods
                      to be compared
            dparams:  dict containing the dimensionality reduction parameters
                      on which to run the grid search
            log_name: name of the output log file containing the score summary

        Returns:
            EstimatorSelection object

        Raises:
            ValueError: Some estimators are missing parameters
        """
        assert exists(dirname(log_name)), 'The path {0} does not exist'.format(log_name)
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
        self.log_name   = log_name
        self.scores     = []

    def fit(self, X, y, cv=5, n_jobs=1, verbose=1, scoring=None, refit=False):
        """Fits each estimation pipeline to the data

        Computes a grid_search for each estimator and dimensionalityreduction 
        combination (for more info check sklearn.model_selection.GridSearchCV).
        
        Args:
            X:       list of data instances
            y:       list of targets
            cv:      number of folds to generate for the stratified cross 
                     validation
            njobs:   number of jobs to run in parallel if possible
            verbose: prints more data while running
            scoring: model evaluation strategy; default is accuracy_score
            refit:   refit an estimator using the best found parameters 
                     on the whole dataset
        """
        
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
        
        logging.basicConfig(filename=self.log_name, level=logging.INFO, format='%(message)s')
        logging.info('Processing files...')
        for dkey in self.dkeys:
            self.grid_s[dkey] = {}
            for mkey in self.mkeys:
                print("Running GridSearchCV for {0} + {1}.".format(dkey, mkey))

                pipe = Pipeline(steps=[
                    ('dim_red', self.dim_r[dkey]), 
                    ('classifier', self.models[mkey])
                ])
                params  = dict(self.mparams[mkey].items() + self.dparams[dkey].items())
                gsearch = GridSearchCV(pipe, params, cv=cv, n_jobs=n_jobs, 
                                  verbose=verbose, scoring=scoring, refit=refit)
                gsearch.fit(X,y)

                for gsc in gsearch.grid_scores_:
                    score = row(mkey, dkey, gsc.cv_validation_scores, gsc.parameters)
                    logging.info('estimator: %s; dim_reduction: %s; min_score: %s; mean_score: %s; max_score: %s; std_score: %s; Parameters: %s', 
                        mkey, dkey, score['min_score'], score['mean_score'], score['max_score'], score['std_score'], gsc.parameters
                        )
                    self.scores.append(score)
                self.grid_s[dkey][mkey] = gsearch

    def score_summary(self, sort_by='mean_score', n=20): 
        """Generates a score summary

        Generates a score summary of all the estimator performances
        
        Args:
            sort_by: str containing the key used to sort the output scores
            n: number of estimator results to include in the summary
        
        Returns:
            pandas dataframe containing the best n scores in terms of sort_by

        """  
        df = (pd.concat(self.scores, axis=1).T)
        df = df.sort_values([sort_by], ascending=False).head(n)
        
        columns = ['model', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]
        return df[columns]

    def get_best_estimator(self, score_sum):
        """Returns the best estimator

        Returns the best estimator and its parameter configuration
        
        Args:
            score_sum: dict containing the score summary of each estimator

        Returns:
            estimator object and name
        """
        mkey = score_sum['model'].iloc[0]
        dkey = score_sum['dim_red'].iloc[0]
        params = self.grid_s[dkey][mkey].best_params_
        logging.info('\nBest estimator: %s, \n Parameters: %s', mkey, params)
        return self.grid_s[dkey][mkey].estimator.set_params(**params)
