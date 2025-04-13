
import io
import os
import re
import shutil
from PIL import Image, ImageCms
from collections import OrderedDict

class FileCleaner:
    """
    A class to clean directory trees, standardize image formats, and track changes.

    Attributes
    ----------
    root_directory : str
        The base directory to clean.
    root_name : str
        The name of the root directory to limit the logging paths.
    output_file : file object
        The file where changes are logged.
    protein_map : dict
        Standardized protein names.
    protein_variations : dict
        Regex patterns for protein name variations.
    """

    def __init__(self, root_directory, output_location, root_name, output_file, control=False):

        self.root_directory = root_directory
        self.output_location = output_location
        self.root_name = root_name
        self.output_file = output_file

        if(control):
            self._specific = OrderedDict({
                "HLA_G": "HLAG5",
                "HLA-G": "HLAG5"
            })
        else:
            self._specific = OrderedDict({
                "HLA_G": "HLAG",
                "HLA-G": "HLAG"
            })

        self._replace_1 = OrderedDict({
            "20-234": "20_34",
            " E ": "_and_",
            "(REPETIÇÃO)": "REPEAT",
            "REPETIÇÃO": "REPEAT",
            "REPETIÇAO": "REPEAT",
            "CD2810x": "CD28 10X",
            "21HCD28": "21H_CD28",
            "2 LÂMINA": "SLIDE_2",
            "2 LÃMINA": "SLIDE_2",
            "LÂMINA 2": "SLIDE_2",      
            "LÃMINA 2": "SLIDE_2",            
            "HE10X": "HE 10X",
            "HE40X": "HE 40X",
            "HLA_G5": "HLAG5",
            "HLA-G G5": "HLAG5",
            "HLA-G (G2)": "HLAG2",
            "HLA-G G2": "HLAG2",
            "HLA_G2": "HLAG2",
            "PD-L1": "PDL1",
            "45PD": "45 PD",
            "PDL-1": "PDL1",
            "P-DL1": "PDL1",
            "PD-1": "PD1",
            "SPIKEIN": "SPIKE",
            " PIKE": "_SPIKE",
            "CTL4": "CTLA4",
            "CTLA-4": "CTLA4",
            "ACE-2": "ACE2",
            "PD-L110X": "PDL1_10X",
            "PD-L14": "PDL1_4",
            "10X .": "10X_0.",
            "40X .": "40X_0.",
            "10X.": "10X_0.",
            "40X.": "40X_0.",
            "10 X .": "10X_0.",
            "40 X .": "40X_0.",
            "410X": "10X",
            "40 X": "40X",
            "10 X": "10X"
        })

        self._replace_2 = OrderedDict({
            "(1)": "_1",
            "(2)": "_2",
            "(3)": "_3",
            "(4)": "_4",
            "(5)": "_5",
            "(6)": "_6",
            "(7)": "_7",
            "(8)": "_8",
            "(9)": "_9",
            "(0)": "_0",
            "(10)": "_10",
            "(11)": "_11",
            "(12)": "_12",
            "(13)": "_13",
            "(14)": "_14",
            "(15)": "_15",
            "(16)": "_16",
            "(17)": "_17",
            "(18)": "_18",
            "(19)": "_19",
            "(20)": "_20",
            "(45)": "_45",
            "(1": "_1",
            "(2": "_2",
            "(3": "_3",
            "(4": "_4",
            "(5": "_5",
            "(6": "_6",
            "(7": "_7",
            "(8": "_8",
            "(9": "_9",
            "(0": "_0",
            "(11": "_11",
            "(12": "_12",
            "(13": "_13",
            "(14": "_14",
            "(15": "_15",
            "(16": "_16",
            "(17": "_17",
            "(18": "_18",
            "(19": "_19",
            "(10": "_10",
            "(20": "_20",
            "1)": "_1",
            "2)": "_2",
            "3)": "_3",
            "4)": "_4",
            "5)": "_5",
            "6)": "_6",
            "7)": "_7",
            "8)": "_8",
            "9)": "_9",
            "0)": "_0",
            "11)": "_11",
            "12)": "_12",
            "13)": "_13",
            "14)": "_14",
            "15)": "_15",
            "16)": "_16",
            "17)": "_17",
            "18)": "_18",
            "19)": "_19",
            "10)": "_10",
            "20)": "_20",
            "-": "_",
            "   ": "_",
            "  ": "_",
            " ": "_",
            ".": "_",
            "(": "_",
            ")": "_"
        })

    def get_path_from_root(self, full_path, root_dir, idx_add=0):
        """
        Extracts the path starting from the specified root directory (including the root).

        Parameters
        ----------
        full_path : str
            The complete file or folder path.
        root_dir : str
            The directory name from which the returned path should start (inclusive).

        Returns
        -------
        str
            The path starting from the root directory. If the root directory is not found,
            returns the original full path.
        """
        # Normalize the paths to avoid OS-related inconsistencies
        full_path = os.path.normpath(full_path)
        
        # Split the path into parts
        parts = full_path.split(os.sep)
        
        # Check if the root_dir exists in the path
        if root_dir in parts:
            idx = parts.index(root_dir)  # Find the index of root_dir
            cleaned_prev = os.path.join(*parts[:idx+idx_add])  # Rebuild path from root_dir backwards
            cleaned_post = os.path.join(*parts[idx+idx_add:])  # Rebuild path from root_dir onwards
            return cleaned_prev, cleaned_post
        else:
            # If root_dir isn't found, return the original path
            return full_path, None

    def process_image_to_tiff(self, old_path, new_path, srgb_profile, towrite_old_file, towrite_new_file, remove_old=True):
        """
        Converts an image to TIFF format with an sRGB color profile.

        Parameters
        ----------
        input_path : str
            Path to the input image.
        new_output_path : str
            Path to save the converted TIFF image.
        srgb_profile : ImageCmsProfile
            The sRGB color profile.
        remove_old : bool, optional
            If True, removes the original image after conversion.
        """
        try:
            image = Image.open(old_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            icc_profile_data = image.info.get("icc_profile", None)
            if icc_profile_data:
                try:
                    input_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile_data))
                except Exception as e:
                    input_profile = srgb_profile
            else:
                input_profile = srgb_profile

            image = ImageCms.profileToProfile(image, input_profile, srgb_profile, outputMode="RGB")
            image.save(new_path, format="TIFF")

            if remove_old:
                self.delete_path(old_path, towrite_old_file, towrite_new_file)
        except Exception as e:
            self.output_file.write(f"# Error processing image {old_path}: {e}\n")

    def standardize_images_to_tiff(self, input_path, towrite_cleaned_file):
        """
        Standardizes supported image formats to TIFF.

        Parameters
        ----------
        input_path : str
            Path to the input image.
        """
        supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ext = os.path.splitext(input_path)[1].lower()

        if ext not in supported_formats:
            return False

        srgb_profile = ImageCms.createProfile("sRGB")
        old_path = os.path.splitext(input_path)[0] + ".jpg"
        new_path = os.path.splitext(input_path)[0] + ".tiff"
        towrite_old_file = os.path.splitext(towrite_cleaned_file)[0] + ".jpg"
        towrite_new_file = os.path.splitext(towrite_cleaned_file)[0] + ".tiff"
        self.process_image_to_tiff(old_path, new_path, srgb_profile, towrite_old_file, towrite_new_file, remove_old=True)
        return True

    def correct_for_existing(self, new_path, existing_filenames):
        """
        Ensures that the provided new_path is unique within the existing_filenames dictionary.
        If a conflict is detected, appends an incrementing suffix (_1, _2, etc.) to the filename.

        Parameters
        ----------
        new_path : str
            The absolute path to the new file, including the filename and extension.
        existing_filenames : dict
            A dictionary with existing file paths as keys and True as values.

        Returns
        -------
        str
            A modified path that is unique within the existing_filenames dictionary.
        """
        
        # If the path doesn't exist in the dictionary, add it and return as-is
        if new_path not in existing_filenames:
            existing_filenames[new_path] = True
            return new_path

        # Split the path into directory, filename, and extension
        dir_path, filename_with_ext = os.path.split(new_path)
        filename, ext = os.path.splitext(filename_with_ext)
        
        # Initialize the suffix counter
        counter = 1
        
        # Iterate until a unique filename is found
        while True:
            new_filename = f"{filename}_{counter}{ext}"
            new_full_path = os.path.join(dir_path, new_filename)
            
            if new_full_path not in existing_filenames:
                existing_filenames[new_full_path] = True
                return new_full_path
            
            counter += 1

    def clean_filename(self, path_name):
        """
        Cleans the filename by applying standard replacements.

        Parameters
        ----------
        path_name : str
            The original filename or directory name.

        Returns
        -------
        str
            Cleaned filename.
        bool
            True if it's an image, False otherwise.
        """

        # Split path into one index inside the root
        new_path_name_before, new_path_name_after_full = self.get_path_from_root(path_name, self.root_name, idx_add=1)
        new_path_name_after, new_path_name_after_ext = os.path.splitext(new_path_name_after_full)

        # Dar upper() no nome do arquivo
        new_path_name_after = new_path_name_after.upper() + "."

        # Removing first vector
        for key, item in self._replace_1.items():
            new_path_name_after = new_path_name_after.replace(key, item)

        # Removing second vector
        for key, item in self._replace_2.items():
            new_path_name_after = new_path_name_after.replace(key, item)

        # Removing specific parts
        for key, item in self._specific.items():
            new_path_name_after = new_path_name_after.replace(key, item)

        # Joining all "_"
        new_path_name_after = re.sub(r'_+', '_', new_path_name_after)

        # Remove trailing "_"
        new_path_name_after = '/'.join(part.strip('_') for part in new_path_name_after.split('/'))

        # Joining path
        new_path_name = "/" + os.path.join(new_path_name_before, new_path_name_after) + new_path_name_after_ext

        try:
            Image.open(path_name).verify()
            return new_path_name, True
        except:
            return new_path_name, False

    def clean_directory_tree(self):
        """
        Cleans the entire directory tree, standardizes images, and logs changes.
        """

        existing_filenames = OrderedDict()

        for dirpath, dirnames, filenames in os.walk(self.root_directory, topdown=False):

            for filename in filenames:

                original_path = os.path.join(dirpath, filename)

                _, after_root_uncleaned_path = self.get_path_from_root(original_path, self.root_name)
                cleaned_original_path, is_image = self.clean_filename(original_path)
                cleaned_original_path = self.correct_for_existing(cleaned_original_path, existing_filenames)
                _, after_root_cleaned_path = self.get_path_from_root(cleaned_original_path, self.root_name)
                cleaned_path = os.path.join(self.output_location, after_root_cleaned_path)

                if is_image:
                    self.create_and_copy(original_path, cleaned_path, after_root_uncleaned_path, after_root_cleaned_path)
                    self.standardize_images_to_tiff(cleaned_path, after_root_cleaned_path)
                else:
                    self.output_file.write("\t".join(["EMPTY", after_root_uncleaned_path, "\n"]))

            """
            for dirname in dirnames:
                original_dirpath = os.path.join(dirpath, dirname)
                after_root_uncleaned_path = os.path.relpath(original_dirpath, self.root_name)
                cleaned_dirname, _ = self.clean_filename(dirname)
                cleaned_dirpath = os.path.join(self.output_location, cleaned_dirname)
                after_root_cleaned_path = os.path.relpath(cleaned_dirpath, self.root_name)

                if not os.listdir(original_dirpath):
                    self.output_file.write("\t".join(["EMPTY", after_root_uncleaned_path, "\n"]))
                else:
                    self.create_and_copy(original_dirpath, cleaned_dirpath, after_root_uncleaned_path, after_root_cleaned_path)
            """

    def create_and_copy(self, original_path, cleaned_path, towrite_uncleaned_path, towrite_cleaned_path):
        """
        Creates necessary directories and copies files to a cleaned path. Logs the changes.

        Parameters
        ----------
        original_path : str
            The original file or directory path.
        cleaned_path : str
            The target path where the file or directory should be copied.
        towrite_uncleaned_path : str
            The original relative path for logging purposes.
        towrite_cleaned_path : str
            The cleaned relative path for logging purposes.
        """

        # Log the changes
        self.output_file.write("\t".join(["COPIED", f"\"{towrite_uncleaned_path}\"", f"\"{towrite_cleaned_path}\"\n"]))

        # Check if the cleaned path is a directory or a file
        if os.path.isdir(original_path):

            # Create directory if it doesn't exist (like `mkdir -p`)
            os.makedirs(cleaned_path, exist_ok=True)

        elif os.path.isfile(original_path):

            # Ensure the destination directory exists
            destination_dir = os.path.dirname(cleaned_path)
            os.makedirs(destination_dir, exist_ok=True)

            # Copy the file to the cleaned path
            shutil.copy(original_path, cleaned_path)
            #shutil.move(original_path, cleaned_path)

        else:
            # Log or handle cases where the original path doesn't exist
            self.output_file.write(f"WARNING: \"{towrite_uncleaned_path}\" does not exist!\n")

    def delete_path(self, input_path, old_output_path, new_output_path):
        """
        Deletes a file and logs the change.

        Parameters
        ----------
        original_file : str
            Path to the file to be deleted.
        towrite_uncleaned_file : str
            The uncleaned relative path for logging.
        towrite_cleaned_file : str
            The cleaned relative path for logging.
        """

        try:
            os.remove(input_path)
            self.output_file.write("\t".join(["DELETED", f"\"{old_output_path}\"", f"\"{new_output_path}\"\n"]))
        except Exception:
            self.output_file.write("\t".join(["FAILED DELETED", f"\"{old_output_path}\"", f"\"{new_output_path}\"\n"]))

    def rename_path(self, original_path, cleaned_path, towrite_uncleaned_path, towrite_cleaned_path):
        """
        Renames a file or directory and logs the change.

        Parameters
        ----------
        original_path : str
            The original file or directory path.
        cleaned_path : str
            The cleaned file or directory path.
        towrite_uncleaned_path : str
            The uncleaned relative path for logging.
        towrite_cleaned_path : str
            The cleaned relative path for logging.
        """
        towrite_uncleaned_path = os.path.join(towrite_uncleaned_path)
        towrite_cleaned_path = os.path.join(towrite_cleaned_path)
        if original_path != cleaned_path:
            self.output_file.write("\t".join(["RENAMED", f"\"{towrite_uncleaned_path}\"", f"\"{towrite_cleaned_path}\"\n"]))
            #os.rename(original_path, cleaned_path)

if __name__ == "__main__":

    # root_name = "DAB_Grupo_Controle"
    # input_file_path = "/Users/egg/Projects/Stainalyzer/data/DAB_Grupo_Controle/"
    # output_file_name = "/Users/egg/Projects/Stainalyzer/data/DAB_Grupo_Controle_Correspondence_1.txt"
    # output_location = "/Users/egg/Projects/Stainalyzer/data/DAB_Grupo_Controle_Clean/"
    root_name = "DAB_IMIP_Tratamento"
    input_file_path = "/Users/egg/Projects/Stainalyzer/data/DAB_IMIP_Tratamento/"
    output_file_name = "/Users/egg/Projects/Stainalyzer/data/DAB_IMIP_Tratamento_Correspondence_1.txt"
    output_location = "/Users/egg/Projects/Stainalyzer/data/DAB_IMIP_Tratamento_Clean/"
    output_file = open(output_file_name, "w")

    # Get the part after root
    cleaner = FileCleaner(input_file_path, output_location, root_name, output_file, control=True)
    cleaner.clean_directory_tree()

    # Close file
    output_file.close()

    """
    root_name = "DAB_IMIP_Tratamento"
    input_file_path = "/Users/egg/Projects/Stainalyzer/data/DAB_IMIP_Tratamento/"
    output_file_name = "/Users/egg/Projects/Stainalyzer/data/DAB_IMIP_Tratamento.txt"
    output_file = open(output_file_name, "w")


    counter = 0
    for dirpath, dirnames, filenames in os.walk(input_file_path, topdown=False):

        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            output_file.write(full_path+"\n")
        #output_file.write(f"{counter}\n")
        #output_file.write(f"{str(dirpath)}\n")
        #output_file.write(f"{str(dirnames)}\n")
        #output_file.write(f"{str(filenames)}\n\n")
        #counter += 1

    output_file.close()
    """




