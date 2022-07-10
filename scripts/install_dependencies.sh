#!/usr/bin/env bash

set -e

if [ "${AUDITWHEEL_PLAT:0:9}" == "manylinux" ]; then
  echo "Installing dependencies for manylinux"
  yum install -y openssl-devel pkgconfig
fi

if [ "${AUDITWHEEL_PLAT:0:9}" == "musllinux" ]; then
  echo "Installing dependencies for musllinux"
  apk add openssl-dev pkgconfig
  apk add --no-cache protoc
fi
