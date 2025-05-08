#!/bin/bash 
#Sherlock_Submission_Test.sh, a sample slurm job  
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling  
#################
#set a job name  
#SBATCH --job-name=lhp1
#################  
#a file for job output, you can check job progress
#SBATCH --output=lhp1_fwselecion_013118.out
#################
# a file for errors from the job
#SBATCH --error=lhp1_fwselecion_013118.err
#################
#time you think you need; default is one hour
#in minutes in this case
#SBATCH --time=1:00:00
#################
#memory per node; default is 4000 MB per CPU
#SBATCH --mem=16000
#################
#quality of service; think of it as job priority
#SBATCH --qos=normal
#################

#now run normal batch commands
module load openmpi/1.6.5/intel13sp1up1

ml matlab


cd $SCRATCH/lhp1
export PATH="$PATH:$SCRATCH/lhp1"
export PATH="$PATH:$SCRATCH/lhp1/bin"

masterPath='/scratch/users/rshe/lhp1/'
savePath='/scratch/users/rshe/lhp1/results/'
time=$1
condition=$2
crossValidation=$3

matlab -nodesktop -nodisplay -nosplash -r "addpath(genpath('/scratch/users/rshe/lhp1/bin/'));lhp1_fwselecion_013118('$masterPath','$savePath','$time','$condition','$crossValidation');exit"



