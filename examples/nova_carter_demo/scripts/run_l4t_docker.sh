#!/bin/bash


docker run \
    --rm \
    -it \
    --name nova_carter_demo \
    --device /dev/snd \
    --network host \
    -v $(pwd):/nova_carter_demo \
    -w /nova_carter_demo \
    nova_carter_demo:l4t
