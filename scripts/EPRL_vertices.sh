#!/bin/sh
#SBATCH -A def-vidotto
#SBATCH -n 100
#SBATCH --cpus-per-task=10
#SBATCH --time=10-0:00:00
#SBATCH --job-name=EPRL_vertex_tensors
#SBATCH --output=EPRL_vertex_tensors.log
#SBATCH --error=EPRL_vertex_tensors.err
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=pfrisoni@uwo.ca

echo "Running on: $SLURM_NODELIST"
echo

# parameters

BASEDIR=/home/frisus95/projects/def-vidotto/frisus95/sl2cfoam_next_updated
DATADIR=/home/frisus95/scratch/sl2cfoam_next_data

export LD_LIBRARY_PATH="${BASEDIR}/lib":$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

IMMIRZI=1.2
SHELLS=20

TspinMin=0
TspinMax=12

# start commands

TCURRENTSPIN=$TspinMin

while [ $TCURRENTSPIN -le $TspinMax ]
do

TJS=$(( TCURRENTSPIN ))

now=$(date)
echo
echo "Starting Lorentzian fulltensor [ TJS = ${TCURRENTSPIN}, shells = ${SHELLS} ]... (now: $now)"

$BASEDIR/bin/vertex-fulltensor -V -h -m 2000 $DATADIR $IMMIRZI $TJS,$TJS,$TJS,$TJS,$TJS,$TJS,$TJS,$TJS,$TJS,$TJS $SHELLS

now=$(date)
echo "... done (now: $now)"
echo

let TCURRENTSPIN=TCURRENTSPIN+1

done

echo
echo "All completed."
