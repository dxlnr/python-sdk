#!/bin/bash

set -e

cd docs
make html
cd build/html
aws s3 sync --delete --exclude ".*" --acl public-read --cache-control "no-cache" ./ s3://docs.modalic.ai
