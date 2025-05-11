https://docs.hpc.cam.ac.uk/hpc/index.html#

### Commands: 

'# login node' 
ssh jaf98@login.hpc.cam.ac.uk

'# navigate to current working directory'
cd rds/hpc-work/geometric-rna-design 

'# reactivate environment'
mamba env list 
-> should now see:
- base                     /home/jaf98/miniforge3
- rna                   *  /home/jaf98/miniforne3/envs/rna

mamba activate rna

**pull the latest updates**!
`cd geometric-rna-design 
'git pull origin main'


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


**When Returning to Project**
1. `conda activate rna`
2. `source .env`
3. Verify paths with `echo $X3DNA`
4. Run `wandb status` before processing
5. Check SLURM queue with `squeue -u jaf98`


###  Comments: 
followed the instructions and see when pip install torch_geometric: 
ERROR: torch 2.5.1 requires sympy == 1.13.1, but you have sympy 1.13.3
pip install sympy == 1.13.1  # Downgrade to exact version PyTorch needs

