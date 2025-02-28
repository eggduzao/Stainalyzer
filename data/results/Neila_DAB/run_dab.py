import os
import sys

class RunPBS:

    def __init__(
        self, prefix, root_name, base_path, input_path, output_path, parameters
    ):
        self.prefix = prefix
        self.root_name = root_name
        self.base_path = base_path
        self.input_path = input_path
        self.output_path = output_path
        self.parameters = parameters

    def create_files(self):

        # Create PBS scripts
        for i, param in enumerate(self.parameters):

            # Parameters
            input_location = os.path.join(self.input_path, param)
            output_location = os.path.join(self.output_path, param)

            # Unique PBS file name
            pbs_filename = f"{self.prefix}_{i}.pbs"

            # Create PBS script
            with open(pbs_filename, "w") as pbs_file:
                pbs_file.write(
                    f"""#!/bin/bash

#PBS -N {self.prefix}_{i}
#PBS -o {self.prefix}_{i}.out
#PBS -e {self.prefix}_{i}.err

#PBS -q workq
# workq - Fila default e sem restrições. Utiliza todos os nós.
# fatq - fila para os fat nodes.
# normq - fila para nodes comuns.
# gpuq - fila para processamento em GPU.
#PBS -V
#PBS -W umask=002

#PBS -l nodes=1:ppn=4
#PBS -l mem=48gb
#PBS -l walltime=12:00:00

# cd $PBS_O_WORKDIR

# Environments
source /sw/miniconda3/bin/activate
conda activate ml

# Current Job Parameter
basepath=\"{self.base_path}\"
input_location=\"{input_location}\"
output_location=\"{output_location}\"
root_name=\"{self.root_name}\"

# Create output path and move to input location
mkdir -p \"$output_location\"
cd $basepath

# Uncompress Control
Stainalyzer --severity 0.5 $input_location $output_location $root_name

"""
                )

    def run_jobs(self):

        # Submit PBS scripts
        for i, param in enumerate(self.parameters):

            # Unique PBS file name
            pbs_filename = f"{self.prefix}_{i}.pbs"

            # Submit job
            os.system(f"qsub {pbs_filename}")

    def merge_files(self):

        # Get name of files
        out_file_list = []
        err_file_list = []
        for i, param in enumerate(self.parameters):

            # Unique out and err PBS file name
            out_filename = f"{self.prefix}_{i}.out"
            err_filename = f"{self.prefix}_{i}.err"

            # Append to list only if it exists
            if os.path.exists(out_filename):
                out_file_list.append(out_filename)
            if os.path.exists(err_filename):
                err_file_list.append(err_filename)

        # Output file name
        out_file_merged = f"{self.prefix}_merged.out"
        err_file_merged = f"{self.prefix}_merged.err"

        # Merge only if list is not empty
        if len(out_file_list) >= 1:
            self._merge(out_file_list, out_file_merged)
        if len(err_file_list) >= 1:
            self._merge(err_file_list, err_file_merged)

    def delete_files(self):

        # Submit PBS scripts
        for i, param in enumerate(self.parameters):

            # Unique PBS, out and err file name
            pbs_filename = f"{self.prefix}_{i}.pbs"
            out_filename = f"{self.prefix}_{i}.out"
            err_filename = f"{self.prefix}_{i}.err"

            # Removing files
            if os.path.exists(pbs_filename):
                os.remove(pbs_filename)
            if os.path.exists(out_filename):
                os.remove(out_filename)
            if os.path.exists(err_filename):
                os.remove(err_filename)

    def _merge(self, file_list, output_file):
        # Merge files
        with open(output_file, "w") as outfile:
            for file in file_list:
                outfile.write(f"> {file}\n")  # Write the filename as a header
                with open(file, "r") as infile:
                    outfile.write(infile.read())  # Append file content
                outfile.write(
                    ("-" * 50) + "\n\n"
                )  # Add 50 dashes and 2 blank lines at the end


if __name__ == "__main__":

    # Parameters
    dataset = sys.argv[1]
    operation = sys.argv[2]

    # Dataset
    base_path = "/storage2/egusmao/projects/Stainalyzer/data/"
    if dataset == "control":
        prefix = "run_control"
        root_name = "DAB_Grupo_Controle"
        input_path = os.path.join(
            base_path, "DAB_Grupo_Controle_Clean/DAB_Grupo_Controle/"
        )
        output_path = os.path.join(base_path, "results/DAB_Grupo_Controle/")
        parameters = [
            "GRUPO_CONTROLE_PE_19_12_2024",
            "GRUPO_CONTROLE_PE_ACE2",
            "GRUPO_CONTROLE_PE_CD28",
            "GRUPO_CONTROLE_PE_CTLA4",
            "GRUPO_CONTROLE_PE_HLAG2",
            "GRUPO_CONTROLE_PE_HLAG5",
            "GRUPO_CONTROLE_PE_PD1",
            "GRUPO_CONTROLE_PE_PDL1",
            "GRUPO_CONTROLE_PE_SPIKE",
        ]
    elif dataset == "treatment":
        prefix = "run_treatment"
        root_name = "DAB_IMIP_Tratamento"
        input_path = os.path.join(
            base_path, "DAB_IMIP_Tratamento_Clean/DAB_IMIP_Tratamento/"
        )
        output_path = os.path.join(base_path, "results/DAB_IMIP_Tratamento/")
        parameters = [
            "IMIP_ACE2",
            "IMIP_CD28",
            "IMIP_CTLA4",
            "IMIP_FOTOS_NOVAS",
            "IMIP_HLAG2",
            "IMIP_HLAG5",
            "IMIP_PD1",
            "IMIP_PDL1",
            "IMIP_SPIKE",
        ]

    # Input handling
    run_pbs = RunPBS(prefix, root_name, base_path, input_path, output_path, parameters)

    # Operation
    if operation == "make":
        run_pbs.create_files()
    elif operation == "run":
        run_pbs.run_jobs()
    elif operation == "merge":
        run_pbs.merge_files()
    elif operation == "del":
        run_pbs.delete_files()
