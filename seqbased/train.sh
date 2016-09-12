#!/bin/sh

# train n networks and write output to file

N=$1
file=$2

echo "Train $N networks, write output to $file"
echo "Train $N networks, write output to $file" > $file

for i in $(seq 1 $N)
do
    echo "Train network $i"
    echo "Train network $i" >>$file
    python train_model.py settings_train >> $file
done
