#!/bin/bash

PREFIX="/Users/egg/Projects/Stainalyzer/data/results/"
SCRIPT_NAME="${PREFIX}compression.sh"

FOLDER_LIST=(
"${PREFIX}Neila_DAB/DAB_Grupo_Controle/GRUPO_CONTROLE_PE_19_12_2024/"
"${PREFIX}Neila_DAB/DAB_Grupo_Controle/GRUPO_CONTROLE_PE_ACE2/"
"${PREFIX}Neila_DAB/DAB_Grupo_Controle/GRUPO_CONTROLE_PE_CD28/"
"${PREFIX}Neila_DAB/DAB_Grupo_Controle/GRUPO_CONTROLE_PE_CTLA4/"
"${PREFIX}Neila_DAB/DAB_Grupo_Controle/GRUPO_CONTROLE_PE_HLAG2/"
"${PREFIX}Neila_DAB/DAB_Grupo_Controle/GRUPO_CONTROLE_PE_HLAG5/"
"${PREFIX}Neila_DAB/DAB_Grupo_Controle/GRUPO_CONTROLE_PE_PD1/"
"${PREFIX}Neila_DAB/DAB_Grupo_Controle/GRUPO_CONTROLE_PE_PDL1/"
"${PREFIX}Neila_DAB/DAB_Grupo_Controle/GRUPO_CONTROLE_PE_SPIKE/"
"${PREFIX}Neila_DAB/DAB_IMIP_Tratamento/IMIP_ACE2/"
"${PREFIX}Neila_DAB/DAB_IMIP_Tratamento/IMIP_CD28/"
"${PREFIX}Neila_DAB/DAB_IMIP_Tratamento/IMIP_CTLA4/"
"${PREFIX}Neila_DAB/DAB_IMIP_Tratamento/IMIP_FOTOS_NOVAS/"
"${PREFIX}Neila_DAB/DAB_IMIP_Tratamento/IMIP_HLAG2/"
"${PREFIX}Neila_DAB/DAB_IMIP_Tratamento/IMIP_HLAG5/"
"${PREFIX}Neila_DAB/DAB_IMIP_Tratamento/IMIP_PD1/"
"${PREFIX}Neila_DAB/DAB_IMIP_Tratamento/IMIP_PDL1/"
"${PREFIX}Neila_DAB/DAB_IMIP_Tratamento/IMIP_SPIKE/"
)

for folder in "${FOLDER_LIST[@]}"; do
    if [ -d "$folder" ]; then
        folder_no_slash="${folder%/}"  # remove trailing slash
        zip_path="${folder_no_slash}.zip"
        echo "Compressing $folder to $zip_path"
        zip -r "$zip_path" "$folder_no_slash"
    else
        echo "⚠️  Folder not found: $folder"
    fi
done

/Users/egg/Projects/Stainalyzer/data/results/Neila_DAB/DAB_Grupo_Controle/GRUPO_CONTROLE_PE_19_12_2024/results.xlsx