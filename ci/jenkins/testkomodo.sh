
copy_test_files () {
    cp -r $CI_SOURCE_ROOT/tests $CI_TEST_ROOT/tests
    TMP_CLONE_DIR=$(mktemp -d)
    git clone https://github.com/equinor/libres.git $TMP_CLONE_DIR/libres
    mv $TMP_CLONE_DIR/libres/test-data $CI_TEST_ROOT/libres-test-data
    export LIBRES_TEST_DATA_DIR=$CI_TEST_ROOT/libres-test-data
}
