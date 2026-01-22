#!/bin/bash
export HOME=/nfs/home/${USER}/.local/bin
export PATH=$HOME/.local/bin:$PATH

#python generate_ECG_data.py 1>main_log.txt 2>&1
python ./cross_validate_datasets.py train 1>main_log.txt 2>&1
