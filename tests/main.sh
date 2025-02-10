#!/bin/bash

INPUT_FOLDER="./DAB_Training_Input/"
OUTPUT_FOLDER="./DAB_Training_Output/"
ROOT_NAME="DAB_Training_Input"

Stainalyzer --severity 0.5 $INPUT_FOLDER $OUTPUT_FOLDER $ROOT_NAME

