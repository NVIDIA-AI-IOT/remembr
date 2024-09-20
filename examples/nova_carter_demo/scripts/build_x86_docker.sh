#!/bin/bash

docker build \
    -t nova_carter_demo:x86 \
    -f $(pwd)/docker/Dockerfile.x86 \
    $(pwd)/docker
