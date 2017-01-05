# Processing arithmetics

This repository contains the code that was used to conduct the analyses presented in the following [workshop paper](http://dieuwkehupkes.nl/research/nips2016.pdf) at NIPS 2016, in which we investigate how recursive and recurrent artificial neural networks can process hierarchical compositionality. We packaged the code, such that it is easy to experiment with it and the results are easily reproducible.

## Installing the package

The processing_arithmetics package uses the following dependencies:

- numpy, scipy, sklearn
- h5py
- matplotlib
- nltk
- theano

Furthermore, the package uses an extended version of Keras, that can be found [here](https://github.com/dieuwkehupkes/keras) to which a few classes and metrics are added. You can install this version of keras by cloning the repository and then doing an editable install via pip:

```sh
git clone https://github.com/dieuwkehupkes/keras
pip install -e /path/to/keras
```

This version of keras is frequently updated to match the master branch of the original keras repository and should behave identically (aside from the added functionality). 
If you already have your own copy of the keras repository on your system, consider adding the keras repository required for the processing_arithmetics package as a remote to your local installation.

You can install the processing_arithmetics package by cloning the repository and doing an editable install via:

```sh
git clone https://github.com/dieuwkehupkes/processing_arithmetics
pip install -e /path/to/processing_arithmetics
```

## Tests

Although the test suite of the package does not cover all functionality, it can be used to confirm if the most important methods still work.
To run the test suite, you will need to install `pytest`: `pip install pytest`. 
Then simply run: `pytest tests/`.

## Scripts

The scripts folder contains a few scripts that can be used to train different kind of models.
