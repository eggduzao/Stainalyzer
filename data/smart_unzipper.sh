#!/bin/bash

# Directory containing the zip files
ZIP_DIR="/Users/egg/Desktop/data-science-bowl-2018/"  # Change if needed

# Loop over each zip file
for zipfile in "$ZIP_DIR"/*.zip; do
    [ -e "$zipfile" ] || continue  # Skip if no zip files

    # Get base name without .zip extension
    basename=$(basename "$zipfile" .zip)
    
    echo "🔍 Checking $zipfile..."

    # Count number of top-level entries in the zip
    num_entries=$(unzip -l "$zipfile" | awk 'NR>3 {print $4}' | grep -v '/$' | cut -d/ -f1 | sort -u | wc -l)

    if [ "$num_entries" -gt 1 ]; then
        echo "📦 Multiple items found. Creating folder: $basename"
        mkdir -p "$basename"
        unzip -q "$zipfile" -d "$basename"
    else
        # One top-level file or one folder? Let's inspect further
        first_entry=$(unzip -l "$zipfile" | awk 'NR>3 {print $4}' | grep -v '/$' | head -n 1)
        if [[ "$first_entry" == */* ]]; then
            echo "📂 Archive has its own folder structure. Extracting to current dir."
            unzip -q "$zipfile"  # Already in a folder
        else
            echo "📁 Single file. Creating folder: $basename"
            mkdir -p "$basename"
            unzip -q "$zipfile" -d "$basename"
        fi
    fi
done

 
# 1. Move all image files to the root directory
#find . -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.tif" -o -iname "*.tiff" \) -exec mv -n {} . \;

# 2. Delete all empty directories (bottom-up)
#find . -type d -empty -delete

# Renamer
# i=1; find . -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.tif" -o -iname "*.tiff" \) | while read -r f; do
#   ext="${f##*.}"
#   newname="image$i.${ext,,}"  # convert extension to lowercase
#   while [[ -e "$newname" ]]; do ((i++)); newname="image$i.${ext,,}"; done
#   mv "$f" "$newname" && ((i++))
# done
