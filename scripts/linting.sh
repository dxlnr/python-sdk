#!/usr/bin/env bash
set -e
set -x

SRC=${1:-"modalic tests examples"}
SRC_NO_TESTS=${1:-"modalic"}

export MODALIC_DEBUG=1
flake8 $SRC
autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place $SRC --exclude=__init__.py --check
isort $SRC scripts --check-only
black $SRC --target-version py38 --check --exclude examples/pytorch_mnist/*.py
# mypy $SRC_NO_TESTS
