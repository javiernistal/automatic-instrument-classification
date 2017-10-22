from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
    AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA, NMF


models = { 
    'ExtraTreesClassifier':       ExtraTreesClassifier(),
    'RandomForestClassifier':     RandomForestClassifier(),
    'AdaBoostClassifier':         AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'SVC':                        SVC()
}

mparams = { 
    'ExtraTreesClassifier':       { 'classifier__n_estimators' : [32, 64] },
    'RandomForestClassifier':     { 'classifier__n_estimators' : [32, 64] },
    'AdaBoostClassifier':         { 'classifier__n_estimators' : [32, 64] },
    'GradientBoostingClassifier': { 'classifier__n_estimators' : [32, 64] , 
                                    'classifier__learning_rate': [0.5]},
    'SVC'                       : {
                                    'classifier__kernel':        ['linear', 'rbf'], 
                                    'classifier__C':             [1000, 10], 
                                    'classifier__gamma':         [1, 0.01]
                                }
}
dim_red = {
    'pca':   PCA(),
    'nmf':   NMF(),
    'kbest': SelectKBest(chi2)
}
dparams = {
    'pca':   {'dim_red__n_components': [30, 40]},
    'nmf':   {'dim_red__n_components': [30, 40]} ,
    'kbest': {'dim_red__k': [30, 40]}
}
preprocess = MinMaxScaler()
norm       = MinMaxScaler()
