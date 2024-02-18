#!/bin/bash
source /workspace/setup.sh
export PYTHONPATH=$PYTHONPATH:$PWD

echo processing: $0 $@
python ./scripts/doStructure.py $@