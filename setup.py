#!/usr/bin/env python
import os
from setuptools import setup


def package_files(directory):
    paths = []
    for (path, _, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


job_files = package_files("semeio/jobs/configs") + package_files("semeio/jobs/scripts")

setup(
    name="semeio",
    version="0.0.1",
    author="Software Innovation Bergen, Equinor ASA",
    description="Jobs and workflow jobs for Ert.",
    packages=[
        "semeio",
        "semeio.hook_implementations",
    ],
    entry_points={
        "ert": [
            "semeio_jobs = semeio.hook_implementations.jobs",
        ]
    },
    install_requires=[],
    test_suite="tests",
    package_data={"semeio": job_configs},
    include_package_data=True,
)
