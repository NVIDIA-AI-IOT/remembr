#!/bin/bash

docker build \
    -t nova_carter_demo:l4t \
    -f $(pwd)/docker/Dockerfile.l4t \
    $(pwd)/docker
