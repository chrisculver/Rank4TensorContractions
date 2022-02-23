#!/bin/bash

export CUTENSOR_LOG_LEVEL=5
export CUTENSOR_LOG_MASK=0
export CUTENSOR_LOG_FILE="cutensor.log"
export CUTENSOR_NVTX_LEVEL=0

./a.out
