rapid-tests:
    OMP_NUM_THREADS=1 pytest -n auto --dist loadgroup tests/ \
    -m "not integration_test"
