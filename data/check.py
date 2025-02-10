import os

def count_files_in_subtree(folder_path):
    """Count total number of files in a directory and all its subdirectories."""
    file_count = 0
    for _, _, files in os.walk(folder_path):
        file_count += len(files)
    return file_count

def list_folders_at_depth(root_path, depth):
    """List folders at a specific depth from the root."""
    folders_at_depth = []
    for dirpath, dirnames, _ in os.walk(root_path):
        current_depth = dirpath[len(root_path):].count(os.sep)
        if current_depth == depth:
            folders_at_depth.append(dirpath)
    return folders_at_depth

def print_file_counts(root_path, depth, label):
    """Print file counts for folders at the specified depth."""
    print(f"\n# {label}:")
    folders = list_folders_at_depth(root_path, depth)
    for folder in sorted(folders):
        total_files = count_files_in_subtree(folder)
        print(f"{folder}\t{total_files}")

if __name__ == "__main__":

    # Define the original and cleaned directory paths
    original_root="/Users/egg/Projects/Stainalyzer/data/DAB_Grupo_Controle/Grupo_Controle_PE_ACE2/"
    cleaned_root="/Users/egg/Projects/Stainalyzer/data/DAB_Grupo_Controle_Clean/DAB_Grupo_Controle/Grupo_Controle_PE_ACE2/"
    #original_root="/Users/egg/Projects/Stainalyzer/data/DAB_IMIP_Tratamento"
    #cleaned_root="/Users/egg/Projects/Stainalyzer/data/DAB_IMIP_Tratamento_Clean/DAB_IMIP_Tratamento"

    # Define the depth level
    depth = 0  # Set to desired depth (0 = root, 1 = first level, etc.)

    # Print file counts for both directories
    print_file_counts(original_root, depth, "Original")
    print_file_counts(cleaned_root, depth, "Cleaned")