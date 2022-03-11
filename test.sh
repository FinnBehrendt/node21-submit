#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
echo $SCRIPTPATH
. ./build.sh

docker volume create noduledetection-output

docker run --rm \
        --network none \
        --gpus all \
        --memory=11g \
        -v $SCRIPTPATH/test/:/input/ \
        -v noduledetection-output:/output/ \
        noduledetector
# docker run --rm \
#         --network none \
#         --gpus all \
#         --memory=11g \
#         -v $SCRIPTPATH/test/:/input/ \
#         -v noduledetection-output:/output/ \
#         noduledetector

# docker run --rm -v noduledetection-output:/output/ python:3.7-slim cat /output/nodules.json | python3 -m json.tool
# echo "inputs"
# docker run --rm -v $SCRIPTPATH/test/:/input/ python:3.7-slim cat /input/expected_output.json | python3 -m json.tool

# docker run --rm \
#         --network none \
#         --gpus all \
#         -v noduledetection-output:/output/ \
#         -v $SCRIPTPATH/test/:/input/ \
#         python:3.7-slim python3 -c "import json, sys; f1 = json.load(open('/output/nodules.json')); f2 = json.load(open('/input/expected_output.json')); print(f1!=f2); sys.exit(f1 != f2);"

# if [ $? -eq 0 ]; then
#     echo "Tests successfully passed..."
# else
#     echo "Expected output was not found..."
# fi

docker volume rm noduledetection-output


