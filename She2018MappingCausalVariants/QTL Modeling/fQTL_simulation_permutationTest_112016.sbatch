#!/bin/bash 
#Sherlock_Submission_Test.sh, a sample slurm job  
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling  
#################
#set a job name  
#SBATCH --job-name=simulation_bootstrap
#################  
#a file for job output, you can check job progress
#SBATCH --output=simulation_bootstrap_112016.out
#################
# a file for errors from the job
#SBATCH --error=simulation_bootstrap_112016.err
#################
#time you think you need; default is one hour
#in minutes in this case
#SBATCH --time=2:00:00
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


cd $SCRATCH/QTL_modeling/simulation_permutationTest/
export PATH="$PATH:$SCRATCH/QTL_modeling/"

Trials=10
cycles=6
genomeSize=12100000
RR=.0000034
pSites=100
noise=0.4
nSites=$1
nSegregants=$2
loopIndex=$3

matlab -nodesktop -nodisplay -nosplash -r "addpath(genpath('/scratch/users/rshe/QTL_modeling/'));addpath(genpath('/scratch/users/rshe/F6cross_glmnet/'));addpath(genpath('/scratch/users/rshe/F6cross_fwselection/'));fQTLmodel_timer_fineMapping_bootstrap_112016('$Trials','$nSites','$cycles','$nSegregants','$genomeSize','$RR','$pSites','$noise','$loopIndex');exit"



