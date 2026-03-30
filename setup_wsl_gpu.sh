#!/bin/bash

# Update and install python3-pip and venv
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

# Create and activate virtual environment
python3 -m venv ~/tf_gpu_venv
source ~/tf_gpu_venv/bin/activate

# Install TensorFlow with GPU support
pip install --upgrade pip
pip install tensorflow[and-cuda]

# Verify GPU
python3 -c "import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))"
