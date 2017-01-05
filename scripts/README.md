# Processing Arithmetics scripts

This directory contains a few scripts that can be used to train/test different types of models, as well as visualise what is going on inside already trained models.

[train_model.py](train_sequential_model.py)
Train a sequential model to interpret sentences from the arithmetic language. Required arguments are the type of trainingsarchitecture (prediction, comparison or sequence to sequence), the type of hidden layer that should be used, the number of epochs to train the model and a filename to which the trained model can be written.

[diagnose_model.py](diagnose_sequential_model.py)
Trains a diagnostic classifier on an already trained sequential model. Run diagnose_model -h for information on usage.

[test_model.py](test_sequential_model.py)
Test an already trained model on a set of predefined subsets of the arithmetic language.

[plot_tree_model.py](plot_tree_model.py)
Plot a treebased model.
