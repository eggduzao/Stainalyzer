import os
import re

def correct_filenames(root_path):
    pattern_correct = re.compile(r'.*(\d{1,2}X)_(\d{1,2})\.tiff$')
    pattern_incorrect = re.compile(r'(.*)(\d{1,2}X)(\d{1,2})\.tiff$')
    
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith('.tiff'):
                correct_match = pattern_correct.match(filename)
                incorrect_match = pattern_incorrect.match(filename)
                
                if not correct_match and incorrect_match:
                    new_filename = f"{incorrect_match.group(1)}_{incorrect_match.group(2)}_{incorrect_match.group(3)}.tiff"
                    old_path = os.path.join(dirpath, filename)
                    new_path = os.path.join(dirpath, new_filename)
                    #os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")

if __name__ == "__main__":
    #input_file_name = "/Users/egg/Projects/Stainalyzer/data/DAB_Grupo_Controle_Clean"
    input_file_name = "/Users/egg/Projects/Stainalyzer/data/DAB_IMIP_Tratamento_Clean"
    correct_filenames(input_file_name)