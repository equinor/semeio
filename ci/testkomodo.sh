
copy_test_files () {
    cp -r $CI_SOURCE_ROOT/tests $CI_TEST_ROOT/tests
    cp $CI_SOURCE_ROOT/pyproject.toml $CI_TEST_ROOT
}

start_tests () {
    pytest --ert-integration
}
