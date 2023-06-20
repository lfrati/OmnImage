#!/bin/bash

for TAR in $(ls tars_groups);
do
  slaunch dggpu --force process.py --source=../winter21_whole --tars=tars_groups/$TAR --dest=feats
done
