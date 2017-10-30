from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
    AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA, NMF
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.mixture import GMM


models = { 
    'ExtraTreesClassifier':       ExtraTreesClassifier(),
    # 'RandomForestClassifier':     RandomForestClassifier(),
    # # 'AdaBoostClassifier':         AdaBoostClassifier(),
    # # 'GradientBoostingClassifier': GradientBoostingClassifier(),
    # 'SVC':                        SVC(),
    # 'KNN':                        knn()
}

mparams = { 
    'ExtraTreesClassifier':       { 'classifier__n_estimators' : [32, 64],
                                    'classifier__class_weight':  [{}, 'balanced'] },
    'RandomForestClassifier':     { 'classifier__n_estimators' : [32, 64],
                                    'classifier__class_weight':  [{}, 'balanced'] },
    'AdaBoostClassifier':         { 'classifier__n_estimators' : [32, 64]},
    'GradientBoostingClassifier': { 'classifier__n_estimators' : [32, 64] , 
                                    'classifier__learning_rate': [0.5, 0.05]},
    'SVC'                       : {
                                    'classifier__kernel':        [
                                                                    # 'linear',   
                                                                    'rbf'
                                                                ], 
                                    'classifier__C':             [100, 10], 
                                    'classifier__gamma':         [1],
                                    'classifier__class_weight':  ['balanced'] 
                                },
    'KNN':                       { 'classifier__n_neighbors':    [2, 3, 5],
                                   'classifier__weights':        ['uniform', 'distance'],
                                   # 'classifier__algorithm':      ['auto'],
                                   # 'classifier__leaf_size':      [30, 10, 40]
                                   }
}


dim_red = {
    'pca':   PCA(),
    # 'nmf':   NMF(),
    # 'kbest': SelectKBest(chi2)
}
dparams = {
    'nmf':   {'dim_red__n_components': [20]},
    'pca':   {'dim_red__n_components': [20 , 33, 43]},
    'kbest': {'dim_red__k':            [20, 33, 43]}
}
