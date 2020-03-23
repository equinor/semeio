Correlated Observations Scaling
===============================

Configuration
-------------

Given a configuration in the format:

.. code-block:: salt

    CALCULATE_KEYS:
        keys:
            - key: FOPR

This configuration will let COS calculate a scaling factor from all data
points in ``FOPR`` and update all data points in ``FOPR``.
You can also specify which indices will be updated on the ``FOPR`` key:

Wildcards
^^^^^^^^^

Keys can be given as wildcards:

.. code-block:: salt

    CALCULATE_KEYS:
        keys:
            - key: WOPR_OP*

This will calculate a scaling factor for all keys matching ``WOPR_OP*`` where
the asteriks will match everything. The patterns are Unix style:

::

    *       matches everything
    ?       matches any single character
    [seq]   matches any character in seq
    [!seq]  matches any char not in seq



Key indices
^^^^^^^^^^^

.. code-block:: salt

    CALCULATE_KEYS:
        keys:
            - key: FOPR
                index: 1-10,50-100

This will calculate the scaling factor from indices 1-10 and 50-100, as
well as update these indices.

If not provided ``UPDATE_KEYS`` will use the same keys configuration as
``CALCULATE_KEYS``. Provided, it allows to specify which keys are to be
scaled:

.. code-block:: salt

    CALCULATE_KEYS:
        keys:
            - key: FOPR
                index: 1-10,50-100
    UPDATE_KEYS:
        keys:
            - key: FOPR
                index: 50-100

This configuration will calculate a scaling factor from indices 1-10,50-100
on ``FOPR``, but only update the scaling on indices ``50-100``.

Configuration for clusters
^^^^^^^^^^^^^^^^^^^^^^^^^^

For clusters (groups) of keys, a list of ``CALCULATE_KEYS`` and
``UPDATE_KEYS`` can be provided (the latter omitted for brevity):

.. code-block:: salt

    -
        CALCULATE_KEYS:
            keys:
                - key: FOPR
    -
        CALCULATE_KEYS:
            keys:
                - key: WOPR_OP1*

This will calculate the scaling factor and do the scaling twice, instead
of passing two different configs.

Additional keywords
^^^^^^^^^^^^^^^^^^^

Scaling factor calculation can be configured with the following keywords

.. code-block:: salt

    CALCULATE_KEYS:
        keys:
            - key: FOPR
                index: 1-10,50-100
        threshold: 0.9
        std_cutoff: 1.0e-5
        alpha: 2.5

``threshold``
    Instructs the job to ignore principal components where the summative
    variance is smaller than this value.

    Optional. Defaults to ``0.95``.

``std_cutoff``
    Filters out any observation that has a standard deviation above this
    value.

    Optional. Defaults to ``1e-6``

``alpha``
    For filtering between ensemble mean and observations.

    Optional. Defaults to ``3``.

.. note::  Between runs on clusters, ``threshold``, ``std_cutoff`` and ``std_cutoff`` are reset.

correlated_observations_scaling
-------------------------------

.. argparse::
   :module: semeio.jobs.scripts.correlated_observations_scaling
   :func: scaling_job_parser
   :prog: correlated_observations_scaling.py
