
copy_test_files () {
    cp -r $CI_SOURCE_ROOT/tests $CI_TEST_ROOT/tests
    TMP_CLONE_DIR=$(mktemp -d)
    # tag should be updated if newest test data should be used
    git clone -b 5.0.1 https://github.com/equinor/libres.git $TMP_CLONE_DIR/libres
    mv $TMP_CLONE_DIR/libres/test-data $CI_TEST_ROOT/libres-test-data
    export LIBRES_TEST_DATA_DIR=$CI_TEST_ROOT/libres-test-data
}

start_tests () {
    pytest --ert_integration
}
