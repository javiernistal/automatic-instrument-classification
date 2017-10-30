===========
Music Instrument Classification
===========

This repository contains the results of a music instrument 
classification task. The code available implements a 
grid-search training/validation pipeline for selecting
the best estimator and dimensionality reduction methods 
based on mean accuracy. The experiments are run over a 
audio feature dataset containing 43 low-level audio features 
for each of 9 instrument classes: bass, guitar, hihat, kick,
piano, saxophone, snare, tom and vocals. 

The program generates a log file containing the performance 
of each estimator grid_search in terms of mean accuracy. 
It is used 5-fold cross validation as evaluation procedure. 
It is also evaluated separately the best model out of the
estimator selection by means of its confusion matrix, accuracy, 
precission, recall and f1-score. All these performance measures 
can be found in the results folder. We can select the parameters 
for the Grid-search in the 'gs_params.py' script. Usage::

    cd [path]/instrument_classifier

    python scripts/main.py -p [path_to_dataset] -n [experiment_name]

    Example:

    python scripts/main.py -p music_group_data -n exp_1


Install
=========

Run the following commands from the root folder:

* easy_install pip (only if you don't have pip installed)

* make init

Running tests (NO TESTS CURRENTLY):
-------------

* pip install nose (if you don't have installed nose)

* make test
