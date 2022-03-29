import os
from setuptools import setup, find_packages

requirements = [
    "numpy>=1.22.3",
    "grpcio>=1.43.0",
    "grpcio-tools>=1.43.0",
    "minio>=1.0.1.1",
    "toml>=0.10.2",
]

extra_requirements = {
    "dev": [
        "pytest>=3.7",
        "black>=22.3.0",
        "tox>=3.24.5",
    ]
}

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()
os.chdir(here)

version = {}
with open(os.path.join(here, "modalic", "_version.py"), encoding="utf-8") as f:
    exec(f.read(), version)

setup(
    name="modalic",
    description="Python SDK library for using the modalic MLOps platform.",
    long_description=long_description,
    version=version["__version__"],
    url="https://github.com/modalic/python-sdk",
    author="Modalic",
    license="MIT",
    install_requires=requirements,
    extras_require=extra_requirements,
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
