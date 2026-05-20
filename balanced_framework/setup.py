from setuptools import setup, find_packages


setup(
    name="antenna-ml-balanced",
    version="0.1.0",
    packages=find_packages(include=("balanced_framework", "balanced_framework.*")),
    install_requires=[
        "numpy>=1.23",
        "pandas>=1.5",
        "matplotlib>=3.6",
        "torch>=2.0",
    ],
)

