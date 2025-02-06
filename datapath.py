
import os
import re
import cv2
import openpyxl
import numpy as np

input_file_path = "/Users/egg/Projects/Stainalyzer/data/DAB_IMIP_Tratamento/"
output_file_path = "/Users/egg/Projects/Stainalyzer/data/datapathimip.txt"
output_excel = "/Users/egg/Projects/Stainalyzer/data/datapath.xls"

output_file = open(output_file_path, "w")
workbook = openpyxl.Workbook()
sheet = workbook.active

for dirpath, dirnames, filenames in os.walk(input_file_path):

    for file in filenames:
        if file.endswith(".jpg"):

            # Get the part after root
            full_path = os.path.join(dirpath, file)
            after_root = full_path.split("DAB_IMIP_Tratamento" + os.sep, 1)[1]

            # Split the remaining path into components
            path_components = after_root.split(os.sep)

            output_file.write("\t".join(path_components)+"\n")

# Save the Excel file
workbook.save(output_excel)

# Close file and excel
workbook.close()
output_file.close()



class FileCleaner:
    """
    Class to clean and standardize file and folder names in a directory tree.
    """

    def __init__(self, root_directory):
        self.root_directory = root_directory
        self.protein_map = {
            "CD28": "CD28",
            "HLA_G5": "HLAG5",
            "HLA_G2": "HLAG2",
            "PD1": "PD1",
            "PDL1": "PDL1",
            "SPIKE": "SPIKE",
            "CTLA4": "CTLA4",
            "ACE2": "ACE2"
        }
        self.protein_variations = {
            r"hl[\s\-_]*g[\s\-_]*5": "HLAG5",
            r"hl[\s\-_]*g[\s\-_]*2": "HLAG2",
            r"pd[\s\-_]*1": "PD1",
            r"pdl[\s\-_]*1": "PDL1",
            r"ctla[\s\-_]*4": "CTLA4",
            r"ace[\s\-_]*2": "ACE2",
            r"spike(in)?": "SPIKE"
        }

    def clean_filename(self, filename, folder_context=""):
        """
        Cleans and standardizes a single filename.

        Parameters
        ----------
        filename : str
            The original filename.
        folder_context : str
            The name of the folder containing the file, used for context-sensitive corrections.

        Returns
        -------
        str
            The cleaned filename.
        """
        # Basic replacements
        replacements = {
            r"[\.\-\(\)\s]+": "_",
            r"[àáâã]": "a",
            r"[éê]": "e",
            r"í": "i",
            r"[óôõ]": "o",
            r"ú": "u",
            r"ç": "c",
            r"repeti[cç][ãa]o": "REPEAT",
            r"la[̂\^]?mina": "SLIDE"
        }

        # Apply basic replacements
        for pattern, repl in replacements.items():
            filename = re.sub(pattern, repl, filename, flags=re.IGNORECASE)

        # Standardize magnification values
        filename = re.sub(r"(\d{2,3})[xX]", lambda m: f"_{m.group(1)[:2]}X_", filename)
        filename = filename.replace("410X", "40X")

        # Correct protein names
        for pattern, correct_name in self.protein_variations.items():
            filename = re.sub(pattern, correct_name, filename, flags=re.IGNORECASE)

        # Special handling for HE and SPIKE
        filename = re.sub(r"\bHE\b", "HE", filename, flags=re.IGNORECASE)
        filename = re.sub(r"\bSPIKE\b", "SPIKE", filename, flags=re.IGNORECASE)

        # Ensure numbering for images without numbers
        if re.search(r"\.(jpg|png|tiff)$", filename, re.IGNORECASE) and not re.search(r"_\d+\.jpg", filename):
            filename = re.sub(r"(\.jpg|\.png|\.tiff)$", r"_01\1", filename, flags=re.IGNORECASE)

        # Remove duplicate underscores
        filename = re.sub(r"_+", "_", filename)
        filename = filename.strip("_")

        return filename

    def clean_directory_tree(self):
        """
        Walks through the directory tree and cleans all filenames and folder names.
        """
        for dirpath, dirnames, filenames in os.walk(self.root_directory, topdown=False):
            # Rename files
            for filename in filenames:
                original_path = os.path.join(dirpath, filename)
                cleaned_filename = self.clean_filename(filename, folder_context=dirpath)
                cleaned_path = os.path.join(dirpath, cleaned_filename)
                if original_path != cleaned_path:
                    os.rename(original_path, cleaned_path)

            # Rename directories
            for dirname in dirnames:
                original_dirpath = os.path.join(dirpath, dirname)
                cleaned_dirname = self.clean_filename(dirname)
                cleaned_dirpath = os.path.join(dirpath, cleaned_dirname)
                if original_dirpath != cleaned_dirpath:
                    os.rename(original_dirpath, cleaned_dirpath)

    def process_files_and_save_to_excel(self, file_extension, output_excel):
        """
        Saves the paths of cleaned files to an Excel spreadsheet.

        Parameters
        ----------
        file_extension : str
            The file extension to look for (e.g., '.tiff').
        output_excel : str
            Path to the output Excel file.
        """
        workbook = openpyxl.Workbook()
        sheet = workbook.active

        for dirpath, dirnames, filenames in os.walk(self.root_directory):
            for filename in filenames:
                if filename.endswith(file_extension):
                    full_path = os.path.join(dirpath, filename)
                    relative_path_parts = os.path.relpath(full_path, self.root_directory).split(os.sep)
                    sheet.append(relative_path_parts)

        workbook.save(output_excel)

