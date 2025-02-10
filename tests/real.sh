#!/bin/bash

INPUT_FOLDER="/Users/egg/Projects/Stainalyzer/data/input/"
OUTPUT_FOLDER="./DAB_Training_Output_Real/"
ROOT_NAME="input"

Stainalyzer --severity 0.5 $INPUT_FOLDER $OUTPUT_FOLDER $ROOT_NAME

