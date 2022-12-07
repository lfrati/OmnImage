#!/bin/bash

set -e
# Any subsequent(*) commands which fail will cause the shell script to exit immediately

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

# Default values
NSAMPLES=${NSAMPLES:="20"}
IMDIR=${IMDIR:="imagenet"}

# extract features for each class
# -> output/feats/<CLASS>
python make/features.py --ims $IMDIR --device cpu

# evolve subset for each class
# -> output/OmnImage_<NSAMPLES>.txt
python make/evolve.py --ims $IMDIR --feats output/feats --samples $NSAMPLES

# based on the evolved subsets extract and resize images from <IMDIR>
# -> output/OmnImage<SIZE>_<NSAMPLES> 
python make/generate.py --ims $IMDIR --subset output/OmnImage_$NSAMPLES.txt

