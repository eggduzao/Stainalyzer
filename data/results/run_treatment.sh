#!/bin/bash

# Parameters
basepath="/storage2/egusmao/projects/Stainalyzer/data/"
input_path=$basepath"/DAB_IMIP_Tratamento_Clean/DAB_IMIP_Tratamento/"
output_path=$basepath"results/DAB_IMIP_Tratamento/"
parameters=("IMIP_ACE2" "IMIP_CD28" "IMIP_CTLA4" "IMIP_FOTOS_NOVAS" "IMIP_HLAG2" "IMIP_HLAG5" "IMIP_PD1" "IMIP_PDL1" "IMIP_SPIKE")

# Get the length of the array
length=${#parameters[@]}

# Loop through the indices
for (( i=0; i<length; i++ )); do

	# Parameters
	input_location=$input_path${parameters[$i]}"/"
	output_location=$output_path${parameters[$i]}"/"
	root_name="DAB_IMIP_Tratamento"

	# Execution
    qsub -v INDEX=$i,BASEPATH="$basepath",INPUT="$input_location",OUTPUT="$output_location",ROOT="$root_name" run_treatment.pbs

done

