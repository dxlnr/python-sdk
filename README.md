<h1 align="center">
  <b>Modalic Python SDK</b><br>
</h1>

<p align="center">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

Python SDK library for the using the Modalic MLOps Federated Learning platform.

## Development
For active development of the library some extra packages are needed
```shell
pip install -e .[dev]
```


More information on how to deploy a python package:
[Packaging Projects with Python](https://packaging.python.org/en/latest/tutorials/packaging-projects/#classifiers)

## Install from source with pip
To install 'modalic' as a package locally run:
```sh
# From the root of this repo
pip install .
```
Type `python` in the terminal and then check the installation by running:
```python
import modalic
print(modalic.__version__)
```

## Developing
### Linting
```sh
pip install black
black modalic/
# Sometimes this does not work, then try
python -m black modalic/
```
### Documentation
```sh
pip install -U sphinx
pip install -U sphinx-book-theme
# Build new documentation
sphinx-build -b html docs/source/ docs/build/html
# Alternatively you could run the following two lines
cd docs
make html

# See docs by opening
# docs/build/html/index.html in the browser or with an IDE extension
```


### Implementation Ideas
- Testing / Parser that checks if the framework architecture is correctly set up and implemented beforehand to avoid python giving run time error.
- Add the server as wrapped dockercontainer within Github.
- Use Poetry
