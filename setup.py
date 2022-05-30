# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['modalic',
 'modalic.client',
 'modalic.client.proto',
 'modalic.client.utils',
 'modalic.config',
 'modalic.data',
 'modalic.logging',
 'modalic.server',
 'modalic.simulation',
 'modalic.storage',
 'modalic.utils']

package_data = \
{'': ['*'], 'modalic': ['proto/*']}

install_requires = \
['grpcio-tools>=1.43.0', 'grpcio>=1.43.0']

setup_kwargs = {
    'name': 'modalic',
    'version': '0.1.0',
    'description': 'Python SDK library for using the modalic MLOps platform.',
    'long_description': '<h1 align="center">\n  <b>Modalic Python SDK</b><br>\n</h1>\n\n<p align="center">\n<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>\n</p>\n\nPython SDK library for the using the Modalic MLOps Federated Learning platform.\n\n## Development\nFor active development of the library some extra packages are needed\n```shell\npip install -e .[dev]\n```\n\n\nMore information on how to deploy a python package:\n[Packaging Projects with Python](https://packaging.python.org/en/latest/tutorials/packaging-projects/#classifiers)\n\n## Install from source with pip\nTo install \'modalic\' as a package locally run:\n```sh\n# From the root of this repo\npip install .\n```\nType `python` in the terminal and then check the installation by running:\n```python\nimport modalic\nprint(modalic.__version__)\n```\n\n## Developing\n### Linting\n```sh\npip install black\nblack modalic/\n# Sometimes this does not work, then try\npython -m black modalic/\n```\n### Documentation\n```sh\npip install -U sphinx\npip install -U sphinx-book-theme\n# Build new documentation\nsphinx-build -b html docs/source/ docs/build/html\n# Alternatively you could run the following two lines\ncd docs\nmake html\n\n# See docs by opening\n# docs/build/html/index.html in the browser or with an IDE extension\n```\n\n\n### Implementation Ideas\n- Testing / Parser that checks if the framework architecture is correctly set up and implemented beforehand to avoid python giving run time error.\n- Add the server as wrapped dockercontainer within Github.\n- Use Poetry\n',
    'author': 'Daniel Illner',
    'author_email': 'daniel.illner@outlook.de',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://modalic.ai',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8,<=3.9.12',
}
from build import *
build(setup_kwargs)

setup(**setup_kwargs)
