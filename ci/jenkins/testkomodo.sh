#!/bin/bash
set -xe
shopt -s extglob
source $KOMODO_ROOT/$RELEASE_NAME/enable

echo "create virtualenv"
ENV=testenv
rm -rf $ENV
mkdir $ENV
python -m virtualenv --system-site-packages $ENV
source $ENV/bin/activate
python -m pip install -r test_requirements.txt

# find and check out the code that was used to build libres for this komodo relase
echo "checkout tag from komodo"
EV=$(cat $KOMODO_ROOT/$RELEASE_NAME/$RELEASE_NAME | grep "semeio:" -A2 | grep "version:")
EV=($EV)    # split the string "version: vX.X.X"
EV=${EV[1]} # extract the version

echo "Using semeio version $EV"
git checkout $EV
rm -rf !("tests"|"$ENV")
echo "running pytest"
python -m pytest \
    --ignore="tests/test_formatting.py"
