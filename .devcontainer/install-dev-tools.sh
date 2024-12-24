#!/bin/sh

sudo apt update && sudo apt install -y libtbb-dev git-lfs
poetry install --with=dev
