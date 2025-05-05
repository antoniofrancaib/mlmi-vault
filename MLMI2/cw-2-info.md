https://docs.hpc.cam.ac.uk/hpc/
pdf: https://www.vle.cam.ac.uk/pluginfile.php/28229311/mod_resource/content/20/MLMI2_practical_v3_0.pdf

Login-Web-Interface: https://docs.hpc.cam.ac.uk/hpc/user-guide/login-web.html?highlight=login%20web

temp: cd /rds/project/rds-xyBFuSj0hm0/MLMI.2024-25/jaf98/MLMI2

**Open the Command Palette** in VSCode (`Ctrl+Shift+P` or `Cmd+Shift+P` on macOS): 
- **Remote-SSH: Connect to Host...** 
	- terminal login: password and TOTP
- **Remote-SSH: Open SSH Configuration File...**
	- already set-up


export X3DNA='/rds/user/jaf98/hpc-work/geometric-rna-design/tools/x3dna-v2.4'

              export PATH='/rds/user/jaf98/hpc-work/geometric-rna-design/tools/x3dna-v2.4/bin':$PATH

          --------------------------------------------------------------

          and then run: source ~/.bashrc

  (4) to verify your installation, type: ./find_pair -h


### Commands: 

login node: 
ssh jaf98@login.hpc.cam.ac.uk

correct working directory: 
cd rds/hpc-work
/MLMI2/exp

activate env (deactivate the current environment by typing source deactivate if it exists):  
source /rds/project/rds-xyBFuSj0hm0/MLMI2.M2024/miniconda3/bin/activate

activate the conda environment  
conda activate py25

submit job: 
sbatch slurm_mlmi2_gpu_baseline

squeue -u jaf98
tail -f logs/out.<JOBID>



notebook: 
ssh jaf98@login.hpc.cam.ac.uk

sintr -A MLMI-jaf98-SL2-CPU -p icelake -N1 -n1 -t 1:0:0 --qos=INTR

source /rds/project/rds-xyBFuSj0hm0/MLMI2.M2024/miniconda3/bin/activate

jupyter notebook --no-browser --ip=* --port=8081

local machine: 
ssh -L 8081:<NODEADDR>:8081 jaf98@login.hpc.cam.ac.uk
Open 
http://127.0.0.1:8081

ssh -L 8081:gpu-q-4:8081 jaf98@login.hpc.cam.ac.uk

--train_json=json/train_fbank_sp.json

request an interactive compute node: 
sintr -A MLMI-jaf98-SL2-CPU -p icelake -N1 -n1 -t 1:0:0 --qos=INTR
or GPU node
sintr -A MLMI-jaf98-SL2-GPU -p ampere --gres=gpu:1 -N1 -n1 -t 1:0:0 --qos=INTR


### **11. General Directory Structure in the Coursework**

**Directories to Know**:
- Shared base directory with original files.
cd /rds/project/rds-xyBFuSj0hm0/MLMI2.M2024 

- `/rds/user/CRSid/hpc-work/MLMI2`: Your personal working directory for the coursework.
- `exp/`: Folder containing scripts and data (e.g., `trainer.py`, `decoder.py`).

### **1. Navigating to the Base Directory**

- **Command**: `cd /rds/project/rds-xyBFuSj0hm0/MLMI2.M2024`

- **Description**: This command takes you to the base directory containing all the coursework scripts, data files, and resources. It's the starting point for accessing the provided files like `run.py`, `trainer.py`, and the `json` files.

---

### **2. Navigating to Your Working Directory**

- **Command**: `cd /rds/user/jaf98/hpc-work/MLMI2`

- **Description**: This command moves you to your own working directory, where you'll copy and modify the code for experiments. This ensures you don't accidentally alter shared files in the base directory.

---

### **3. Copying Code to the Working Directory**

- **Command**: 
`cp -r /rds/project/rds-xyBFuSj0hm0/MLMI2.M2024/exp ./ cd exp/`

- **Description**:
    - The `cp -r` command copies the `exp/` folder (containing the coursework code) from the base directory to your working directory.
    - The `cd exp/` command navigates into the `exp/` folder within your working directory, where you'll execute scripts and manage code modifications.

---
### **7. Navigating Within Interactive Nodes**

- **Command**: `cd ~ cd /rds/user/jaf98/hpc-work/MLMI2`
    
- **Description**: Once an interactive node is allocated, you will be redirected to your home directory. Use the `cd` command to navigate back to your working directory.

---

### **8. Activating the Environment**

- **Command**: Before running scripts:
`cd /rds/user/CRSid/hpc-work/MLMI2/exp source /rds/project/rds-xyBFuSj0hm0/MLMI2.M2024/miniconda3/bin/activate conda activate py25`

- **Description**:
    - The `cd` command navigates into the `exp/` folder where the scripts are located.
    - The `source` and `conda activate` commands activate the Anaconda virtual environment needed to execute the Python scripts.

---

### **9. Navigating for Jupyter Notebook Setup**

- **Command**:
`cd /rds/user/jaf98/hpc-work/MLMI2`

- **Description**: Before starting a Jupyter Notebook session on an allocated interactive node, navigate to your working directory to access the notebook and associated scripts.

---

### **10. Navigating Between Nodes**

- **Command**: To move between login nodes:
`ssh login-q-1`

Or from inside an active session:
`ssh login-p-1`

- **Description**: Use SSH commands to move between nodes without navigating from the local machine. This is useful for accessing specific computational resources.

---

