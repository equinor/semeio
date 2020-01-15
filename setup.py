#!/usr/bin/env python
import os
from glob import glob
from setuptools import setup, find_packages


def package_files(directory):
    paths = []
    for (path, _, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


job_files = package_files("semeio/jobs/config_jobs") + package_files(
    "semeio/jobs/config_workflow_jobs"
)
scripts = glob("semeio/jobs/scripts/[!__]*.py")

setup(
    name="semeio",
    use_scm_version={"write_to": "semeio/version.py"},
    author="Software Innovation Bergen, Equinor ASA",
    author_email="fg_gpl@statoil.com",
    url="https://github.com/equinor/semeio",
    description="Jobs and workflow jobs for Ert.",
    packages=find_packages(include=["semeio*"], exclude=["semeio.jobs.scripts"]),
    entry_points={
        "ert": [
            "semeio_jobs = semeio.hook_implementations.jobs",
            "SpearmanCorrelation = semeio.workflows.spearman_correlation_job.spearman_correlation",  # noqa
            "MisfitPreprocessor = semeio.workflows.misfit_preprocessor.misfit_preprocessor",  # noqa
            "CorrelatedObsScaling = semeio.workflows.correlated_observations_scaling.cos",  # noqa
            "CsvExport2Job = semeio.jobs.scripts.csv_export2",
        ]
    },
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=[
        "configsuite>=0.6",
        "numpy",
        "pandas",
        "scipy",
        "xlrd",
        "stea",
        "pyscal>=0.4.0",
        "fmu-ensemble",
        "segyio",
    ],
    setup_requires=["setuptools_scm"],
    package_data={"": job_files},
    scripts=scripts,
    include_package_data=True,
)
