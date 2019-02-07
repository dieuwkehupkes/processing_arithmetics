#!/bin/bash

mkdir -p diagnoses
mkdir -p gate_diagnoses

METRICS="accuracy mean_squared_error"
CLASSIFIERS="subtracting intermediate_locally intermediate_recursively grammatical depth minus1depth minus2depth minus3depth minus4depth minus1depth_count switch_mode"
TEST_CLASSIFIERS="subtracting intermediate_locally intermediate_recursively grammatical depth minus1depth minus2depth minus3depth minus4depth"

RUN=$1
LOOPS=$2
HIDDEN="GRU"
FORMAT="infix"
ARCH="ScalarPrediction"
MODELNAME="${ARCH}_${HIDDEN}_${FORMAT}_${RUN}"

echo "Buiding $LOOPS models for $MODELNAME ..."
python ../scripts/train_sequential_model.py -N $LOOPS -architecture $ARCH --hidden $HIDDEN --nb_epochs 800 --format $FORMAT --visualise_embeddings --save_to $MODELNAME --test

for n in $(seq 0 $(($LOOPS - 1))); do
#for n in 10 16; do
    dc_name="${MODELNAME}_${n}"
    echo Bulding Diagnostic classifier $n for $MODELNAME ...
    python ../scripts/diagnose_sequential_model.py -models "${dc_name}.h5" --nb_epochs 100 --save_to "${dc_name}_diagnosis" --target_folder diagnoses -classifiers $CLASSIFIERS
    echo Bulding Diagnostic gate classifier $n for $MODELNAME ...
    python ../scripts/diagnose_sequential_model.py --test_gates -models "${dc_name}.h5" --nb_epochs 100 --save_to "${dc_name}_diagnosis" --target_folder gate_diagnoses -classifiers $GATE_CLASSIFIERS

    echo Testing Diagnostic classifier for $dc_name ...
    python ../scripts/test_sequential_model.py -architecture DiagnosticClassifier -models "diagnoses/${dc_name}_dc8.h5" -metrics $METRICS -classifiers $TEST_CLASSIFIERS -save_to "${dc_name}_DCresult.pkl"
    echo Testing Diagnostic gate classifier for $dc_name ...
    python ../scripts/test_sequential_model.py -architecture DCgates -models "gate_diagnoses/${dc_name}_dc8.h5" -metrics $METRICS -classifiers $GATE_CLASSIFIERS -save_to "${dc_name}_gate_DCresult.pkl"
    done
