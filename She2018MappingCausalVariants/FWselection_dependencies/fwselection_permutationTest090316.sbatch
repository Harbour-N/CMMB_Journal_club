#!/bin/bash 
#Sherlock_Submission_Test.sh, a sample slurm job  
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling  
#################
#set a job name  
#SBATCH --job-name=permTest
#################  
#a file for job output, you can check job progress
#SBATCH --output=permutationTest.out
#################
# a file for errors from the job
#SBATCH --error=permutationTest.err
#################
#time you think you need; default is one hour
#in minutes in this case
#SBATCH --time=20:00:00
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


cd $SCRATCH/F6cross_fwselection
export PATH="$PATH:$SCRATCH/F6cross_fwselection"

masterPath='/scratch/users/rshe/F6cross_fwselection/'
savePath='/scratch/users/rshe/F6cross_fwselection/permutationTests/'
pcutoff=0.01
row=$1
pIndex=$2


matlab -nodesktop -nodisplay -nosplash -r "addpath(genpath('/scratch/users/rshe/F6cross_fwselection/'));fwselection_permutationTest_trimmedGenotype_081916('$masterPath','$savePath','$pcutoff','$row','$pIndex');exit"



