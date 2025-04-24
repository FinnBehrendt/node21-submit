#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

# docker build -t noduledetector "$SCRIPTPATH"
docker buildx build --platform linux/amd64 -t noduledetector "$SCRIPTPATH"
