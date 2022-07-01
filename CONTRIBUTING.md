# Contributing to Modalic

As an open platform for developers, engineers, researchers and scientists,

## Development

Modalic's Python SDK is some external dependencies. Installing all core dev-dependencies is done by:

```
pip install -r requirements.txt
```

For ensuring code formatting and linting, run the following scripts as .sh:
```
bash scripts/linting.sh
```

For testing the code base, run the following script as .sh:
```
bash scripts/testing.sh
```

### Building the Documentation

To build documentation in various formats, [Sphinx](http://www.sphinx-doc.org) is needed.

```shell
# Auto generate the package module.
cd docs && sphinx-apidoc -o source/sdk ../modalic
# Build new documentation
sphinx-build -b html docs/source/ docs/build/html
# Alternatively you could run the following two lines
cd docs && make html

# See docs by opening
# docs/build/html/index.html in the browser or with an IDE extension
```

##  Code of Conduct
