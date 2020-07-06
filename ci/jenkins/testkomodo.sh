#!/bin/bash
set -xe
shopt -s extglob
PROJECT=semeio
RELEASE_PATH=${KOMODO_ROOT}/${RELEASE_NAME}
GIT=${SDPSOFT}/bin/git
source $KOMODO_ROOT/$RELEASE_NAME/enable
export LD_LIBRARY_PATH=${RELEASE_PATH}/root/lib:${RELEASE_PATH}/root/lib64

echo "fetch libres test data"
$GIT clone https://github.com/equinor/libres.git

echo "create virtualenv"
ENV=testenv
rm -rf $ENV
mkdir $ENV
python -m virtualenv --system-site-packages $ENV
source $ENV/bin/activate
python -m pip install -r test_requirements.txt
ROOT_DIR=$(pwd)
if [[ -z "${sha1// }" ]]; then
    EV=$(cat ${RELEASE_PATH}/${RELEASE_NAME} | grep "${PROJECT}:" -A2 | grep "version:")
    EV=($EV)    # split the string "version: vX.X.X"
    EV=${EV[1]} # extract the version
    EV=${EV%"+py3"}
    echo "Using ${PROJECT} version ${EV}"
    $GIT checkout $EV
    mkdir temp_tests
    mv tests temp_tests

    pushd temp_tests
fi

echo "running pytest"
LIBRES_TEST_DATA_DIR=$ROOT_DIR/libres/test-data python -m pytest tests
