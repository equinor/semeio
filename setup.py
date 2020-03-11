#!/usr/bin/env python
import os
from setuptools import setup


def package_files(directory):
    paths = []
    for (path, _, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


job_files = (
    package_files("semeio/jobs/config_jobs")
    + package_files("semeio/jobs/config_workflow_jobs")
    + package_files("semeio/jobs/scripts")
)

setup(
    name="semeio",
    use_scm_version={"write_to": "semeio/version.py"},
    author="Software Innovation Bergen, Equinor ASA",
    author_email="fg_gpl@statoil.com",
    url="https://github.com/equinor/semeio",
    description="Jobs and workflow jobs for Ert.",
    packages=[
        "semeio",
        "semeio.hook_implementations",
        "semeio.jobs.correlated_observations_scaling",
        "semeio.jobs.design2params",
    ],
    entry_points={"ert": ["semeio_jobs = semeio.hook_implementations.jobs"]},
    license="GPL-3.0",
    platforms="any",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Plugins",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=[
        "configsuite",
        "numpy",
        "pandas",
        "six>=1.12.0",
        "scipy",
        "xlrd",
        "stea",
        "pyscal>=0.4.0",
    ],
    setup_requires=["pytest-runner", "setuptools_scm"],
    tests_require=[
        "pytest",
        "mock",
        'openpyxl<=2.6.4; python_version<="2.7"',
        'openpyxl; python_version>="3.6"',
        'black; python_version>="3.6"',
    ],
    test_suite="tests",
    package_data={"": job_files},
    include_package_data=True,
)
