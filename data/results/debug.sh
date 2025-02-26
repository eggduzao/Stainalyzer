#!/bin/bash

# Parameters
basepath="/storage2/egusmao/projects/Stainalyzer/data/"
input_path=$basepath"/DAB_IMIP_Tratamento_Clean/DAB_IMIP_Tratamento/"
output_path=$basepath"results/DAB_IMIP_Tratamento/"

# Parameters
input_location=$input_path${parameters[$i]}"/"
output_location=$output_path${parameters[$i]}"/"
root_name="DAB_IMIP_Tratamento"

# Execution
qsub -v BASEPATH="$basepath",INPUT="$input_location",OUTPUT="$output_location",ROOT="$root_name" debug.pbs

