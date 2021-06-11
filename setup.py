from setuptools import setup, find_packages


setup(
    name="semeio",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    use_scm_version={"write_to": "semeio/version.py"},
    author="Equinor ASA",
    author_email="fg_sib-scout@equinor.com",
    url="https://github.com/equinor/semeio",
    description="Jobs and workflow jobs for Ert.",
    packages=find_packages(include=["semeio*"]),
    entry_points={
        "ert": [
            "semeio_jobs = semeio.hook_implementations.jobs",
            "SpearmanCorrelation = semeio.workflows.spearman_correlation_job.spearman_correlation",  # noqa
            "MisfitPreprocessor = semeio.workflows.misfit_preprocessor.misfit_preprocessor",  # noqa
            "CorrelatedObsScaling = semeio.workflows.correlated_observations_scaling.cos",  # noqa
            "CsvExport2Job = semeio.workflows.csv_export2.csv_export2",
            "AhmAnalysisJob = semeio.workflows.ahm_analysis.ahmanalysis",
        ],
        "console_scripts": [
            "csv_export2=semeio.workflows.csv_export2.csv_export2:cli",
            "overburden_timeshift=semeio.jobs.scripts.overburden_timeshift:main_entry_point",  # noqa
            "design2params=semeio.jobs.scripts.design2params:main_entry_point",
            "gendata_rft=semeio.jobs.scripts.gendata_rft:main_entry_point",
            "design_kw=semeio.jobs.scripts.design_kw:main_entry_point",
            "fm_pyscal=semeio.jobs.scripts.fm_pyscal:main_entry_point",
            "semeio_stea=semeio.jobs.scripts.fm_stea:main_entry_point",
        ],
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
        "ecl",
        "ert<2.25",
        "equinor-libres",
        "configsuite>=0.6",
        "numpy",
        "pandas>0.24",
        "pydantic",
        "scikit-learn",
        "scipy",
        "xlrd<2",
        "stea",
        "pyscal>=0.4.0",
        "fmu-ensemble",
        "segyio",
        "typing_extensions; python_version < '3.8'",
        "xtgeo",
        "typing_extensions; python_version < '3.8'",
    ],
    include_package_data=True,
)
