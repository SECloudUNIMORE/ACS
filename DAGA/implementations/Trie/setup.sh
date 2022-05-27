#! /bin/bash

set -ue
cd c-code
./compile.sh 2
cd -
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install tqdm
