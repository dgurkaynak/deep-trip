#!/bin/sh

# By using curl
curl -O https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip && unzip -n inception5h.zip && rm inception5h.zip

# or by using wget
# wget https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip && unzip -n inception5h.zip && rm inception5h.zip
