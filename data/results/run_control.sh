#!/bin/bash

# Parameters
basepath="/storage2/egusmao/projects/Stainalyzer/data/"
input_path=$basepath"/DAB_Grupo_Controle_Clean/DAB_Grupo_Controle/"
output_path=$basepath"results/DAB_Grupo_Controle/"
parameters=("GRUPO_CONTROLE_PE_19_12_2024" "GRUPO_CONTROLE_PE_ACE2" "GRUPO_CONTROLE_PE_CD28" "GRUPO_CONTROLE_PE_CTLA4" "GRUPO_CONTROLE_PE_HLAG2" "GRUPO_CONTROLE_PE_HLAG5" "GRUPO_CONTROLE_PE_PD1" "GRUPO_CONTROLE_PE_PDL1" "GRUPO_CONTROLE_PE_SPIKE")

# Get the length of the array
length=${#parameters[@]}

# Loop through the indices
for (( i=0; i<length; i++ )); do

	# Parameters
	input_location=$input_path${parameters[$i]}"/"
	output_location=$output_path${parameters[$i]}"/"
	root_name="DAB_Grupo_Controle"

	# Execution
    qsub -v INDEX=$i,BASEPATH="$basepath",INPUT="$input_location",OUTPUT="$output_location",ROOT="$root_name" run_control.pbs

done

