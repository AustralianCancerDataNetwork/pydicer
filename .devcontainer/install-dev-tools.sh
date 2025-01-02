#!/bin/sh

sudo apt update && sudo apt install -y libtbb-dev
poetry install --with=dev
