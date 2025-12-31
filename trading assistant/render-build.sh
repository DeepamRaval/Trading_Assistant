#!/bin/bash

# Install build tools
apt-get update
apt-get install -y g++ build-essential

# Compile C++ engine
cd cpp
g++ -shared -fPIC -o engine.so engine.cpp
cd ..

# Install Python dependencies
pip install -r requirements.txt
