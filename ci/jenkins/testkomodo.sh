#!/bin/bash
set -xe
shopt -s extglob
PROJECT=semeio
RELEASE_PATH=${KOMODO_ROOT}/${RELEASE_NAME}
GIT=${SDPSOFT}/bin/git
source $KOMODO_ROOT/$RELEASE_NAME/enable

echo "create virtualenv"
ENV=testenv
rm -rf $ENV
mkdir $ENV
python -m virtualenv --system-site-packages $ENV
source $ENV/bin/activate
python -m pip install -r test_requirements.txt

if [[ -z "${sha1// }" ]]; then
    EV=$(cat ${RELEASE_PATH}/${RELEASE_NAME} | grep "${PROJECT}:" -A2 | grep "version:")
    EV=($EV)    # split the string "version: vX.X.X"
    EV=${EV[1]} # extract the version
    EV=${EV%"+py3"}
    echo "Using ${PROJECT} version ${EV}"
    $GIT checkout $EV

    rm -rf !("tests"|"$ENV")
fi

echo "running pytest"
python -m pytest \
    --ignore="tests/test_formatting.py"
