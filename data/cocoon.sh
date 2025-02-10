#!/bin/bash

# Usage:
# ./cocoon.sh compress ./DAB_Grupo_Controle_Clean
# ./cocoon.sh compress ./DAB_IMIP_Tratamento_Clean
# ./cocoon.sh decompress ./DAB_Grupo_Controle_Clean.tar.gz
# ./cocoon.sh decompress ./DAB_IMIP_Tratamento_Clean.tar.gz

# Variables
MODE=$1            # 'compress' or 'decompress'
TARGET_PATH=$2     # Path to the folder (for compression) or tar.gz file (for decompression)

# Check if MODE and TARGET_PATH are provided
if [ -z "$MODE" ] || [ -z "$TARGET_PATH" ]; then
    echo "Usage: $0 [compress|decompress] [path_to_folder_or_tar.gz_file]"
    exit 1
fi

# Resolve absolute path (handles both relative and absolute paths)
ABS_PATH=$(realpath "$TARGET_PATH")

# Function to compress a directory into tar.gz
compress_directory() {
    if [ ! -d "$ABS_PATH" ]; then
        echo "Error: '$ABS_PATH' is not a directory."
        exit 1
    fi
    
    # Extract folder name and create tar.gz file
    FOLDER_NAME=$(basename "$ABS_PATH")
    PARENT_DIR=$(dirname "$ABS_PATH")
    
    tar -czvf "$PARENT_DIR/${FOLDER_NAME}.tar.gz" -C "$PARENT_DIR" "$FOLDER_NAME"
    echo "Compression complete: $PARENT_DIR/${FOLDER_NAME}.tar.gz"
}

# Function to decompress a tar.gz file
decompress_archive() {
    if [ ! -f "$ABS_PATH" ]; then
        echo "Error: '$ABS_PATH' is not a file."
        exit 1
    fi

    if [[ "$ABS_PATH" != *.tar.gz ]]; then
        echo "Error: '$ABS_PATH' is not a .tar.gz file."
        exit 1
    fi
    
    # Extract to the same directory where the tar.gz file is located
    DEST_DIR=$(dirname "$ABS_PATH")
    
    tar -xzvf "$ABS_PATH" -C "$DEST_DIR"
    echo "Decompression complete: Extracted in $DEST_DIR"
}

# Mode Selection
case "$MODE" in
    compress)
        compress_directory
        ;;
    decompress)
        decompress_archive
        ;;
    *)
        echo "Invalid mode. Use 'compress' or 'decompress'."
        exit 1
        ;;
esac

