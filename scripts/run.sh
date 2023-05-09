#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-08:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=jwogan2@uwo.ca

module load cuda cudnn 
source tensorflow/bin/activate

python ./test-parameters.py