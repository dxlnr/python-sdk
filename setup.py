#!/usr/bin/env python
import platform
import sys
import os

from setuptools import setup

from setuptools_rust import Binding, RustExtension

from tools.prebuild import build_deps


python_min_version = (3, 8, 0)
python_min_version_str = ".".join(map(str, python_min_version))
if sys.version_info < python_min_version:
    print(
        "You are using Python {}. Python >={} is required.".format(
            platform.python_version(), python_min_version_str
        )
    )
    sys.exit(-1)

# Force the wheel to be platform specific
# https://stackoverflow.com/a/45150383/3549270
# There's also the much more concise solution in
# https://stackoverflow.com/a/53463910/3549270,
# but that would requires python-dev
try:
    # noinspection PyPackageRequirements,PyUnresolvedReferences
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    # noinspection PyPep8Naming,PyAttributeOutsideInit
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False

except ImportError:
    bdist_wheel = None

# Project root.
cwd = os.path.dirname(os.path.abspath(__file__))


def main():
    # prebuilding dependencies
    build_deps(cwd)

    setup(
        install_requires=[
            "numpy>=1.23.0",
            "grpcio>=1.43.0",
            "grpcio-tools>=1.43.0",
            "protobuf==3.20",
            "toml>=0.10.2",
            "backoff>=2.2.1",
        ],
        rust_extensions=[
            RustExtension(
                {"aggregator": "modalic.bin.aggregator"},
                "modules/mosaic/mosaic/Cargo.toml",
                binding=Binding.Exec,
            )
        ],
    )


if __name__ == "__main__":
    main()
