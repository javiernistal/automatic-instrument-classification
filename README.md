===========
Music Instrument Classification
===========

This repository contains the results of a music instrument 
classification task. The code available implements a 
grid-search training/validation pipeline for selecting
the best estimator and dimensionality reduction methods 
based on mean accuracy. We can select the parameters 
for the Grid-search in the 'instrument_classification/gs_params.py' 
script. 

For the task we attempt on automatic instrument classification, 
the experiments are run over an audio feature dataset 
containing 43 low-level audio features 
for each of 9 instrument classes: bass, guitar, hihat, kick,
piano, saxophone, snare, tom and vocals. We do not provide 
the dataset due to ownership rights. However, if you have 
your own feature-dataset, you can still use the
estimator-selection pipeline to train the best model for
any supervised classification task.

The program generates a log file containing the performance 
of each estimator grid_search in terms of mean accuracy. 
It is used a 5-fold cross validation as evaluation procedure. 
It is also evaluated separately the best model out of the
estimator selection by means of its confusion matrix, accuracy, 
precission, recall and f1-score. All these performance metrics 
can be found in the results folder. Usage::

    cd [path]/automatic-instrument-classification

    python automatic-instrument-classification/main.py -p [path_to_dataset] -n [experiment_name]

    Example:

    python automatic-instrument-classification/main.py -p music_group_data -n exp_1


Install
=========

Run the following commands from the root folder:

* easy_install pip (only if you don't have pip installed)

* sudo make init

If you want to install the estimator_selection pipeline as a package:

* sudo make install

Running tests (NO TESTS CURRENTLY):
-------------

* pip install nose (if you don't have installed nose)

* make test
