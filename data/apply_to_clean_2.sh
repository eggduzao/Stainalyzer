#!/usr/bin/env bash

# Set the root directory where renaming should start
#ROOT_DIR="/Users/egg/Projects/Stainalyzer/data/DAB_Grupo_Controle_Clean/DAB_Grupo_Controle/"
ROOT_DIR="/Users/egg/Projects/Stainalyzer/data/DAB_IMIP_Tratamento_Clean/DAB_IMIP_Tratamento/"

# Use find to search for all files recursively
find "$ROOT_DIR" -type f | while read -r file; do
    new_file="$file"

    # Apply substitutions
    new_file="${new_file//_SPIKE10X_/_SPIKE_10X_}"
    new_file="${new_file//_SPIKE40X_/_SPIKE_40X_}"
    new_file="${new_file//_CD2810X_/_CD28_10X_}"
    new_file="${new_file//_CD2840X_/_CD28_40X_}"
    new_file="${new_file//_PDL110X_/_PDL1_10X_}"
    new_file="${new_file//_PDL140X_/_PDL1_40X_}"
    new_file="${new_file//_HLAG210X_/_HLAG2_10X_}"
    new_file="${new_file//_HLAG240X_/_HLAG2_40X_}"

    # If the new filename differs from the original, rename the file
    if [[ "$new_file" != "$file" ]]; then
        mv "$file" "$new_file"
        echo "Renamed: $file -> $new_file"
    fi
done

