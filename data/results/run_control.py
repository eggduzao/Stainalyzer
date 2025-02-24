
import os
import sys

class RunPBS:

	def __init__(self, root_name, base_path, input_path, output_path, parameters):
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
		    pbs_filename = f"run_control_{i}.pbs"

		    # Create PBS script
		    with open(pbs_filename, "w") as pbs_file:
		        pbs_file.write(f"""#!/bin/bash
					#!/bin/bash

					#PBS -N run_control_{i}
					#PBS -o run_control_{i}.out
					#PBS -e run_control_{i}.err

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
					source $HOME"/env/bin/activate"

					# Ensure OpenGL libraries are accessible
					export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH"

					# Current Job Parameter
					basepath={self.base_path}
					input_location={input_location}
					output_location={output_location}
					root_name={self.root_name}

					# Create output path and move to input location
					mkdir -p $output_location
					cd $basepath

					# Uncompress Control
					Stainalyzer --severity 0.5 $input_location $output_location $root_name

				""")

	def run_jobs(self):	    

		# Submit PBS scripts
		for i, param in enumerate(self.parameters):

		    # Unique PBS file name
		    pbs_filename = f"run_control_{i}.pbs"

	    	# Submit job
	    	os.system(f"qsub {pbs_filename}")

if __name__ == "__main__":

	# Parameters
	root_name = "DAB_Grupo_Controle"
	base_path = "/storage2/egusmao/projects/Stainalyzer/data/"
	input_path = os.path.join(base_path, "DAB_Grupo_Controle_Clean/DAB_Grupo_Controle/")
	output_path = os.path.join(base_path, "results/DAB_Grupo_Controle/")
	parameters = [
	    "GRUPO_CONTROLE_PE_19_12_2024", "GRUPO_CONTROLE_PE_ACE2", "GRUPO_CONTROLE_PE_CD28",
	    "GRUPO_CONTROLE_PE_CTLA4", "GRUPO_CONTROLE_PE_HLAG2", "GRUPO_CONTROLE_PE_HLAG5",
	    "GRUPO_CONTROLE_PE_PD1", "GRUPO_CONTROLE_PE_PDL1", "GRUPO_CONTROLE_PE_SPIKE"
	]

	# Input handling
	run_pbs = RunPBS(root_name, base_path, input_path, output_path, parameters)
	if(sys.arg[1] == "make"):
		run_pbs.create_files()
	elif(sys.arg[1] == "run"):
		run_pbs.run_jobs()


