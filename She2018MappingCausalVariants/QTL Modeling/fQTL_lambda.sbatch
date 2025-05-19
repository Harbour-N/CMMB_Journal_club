#!/bin/bash 
#Sherlock_Submission_Test.sh, a sample slurm job  
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling  
#################
#set a job name  
#SBATCH --job-name=QTLmodel
#################  
#a file for job output, you can check job progress
#SBATCH --output=lambdaQTLmodel.out
#################
# a file for errors from the job
#SBATCH --error=lambdaQTLmodel.err
#################
#time you think you need; default is one hour
#in minutes in this case
#SBATCH --time=40:00:00
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


cd $SCRATCH/QTL_modeling/
export PATH="$PATH:$SCRATCH/QTL_modeling/"

Trials=100
cycles=6
genomeSize=12000000
RR=.000005
pSites=60
noise=0.5
nSites=10000
nSegregants=1100
lambda=$1


matlab -nodesktop -nodisplay -nosplash -r "addpath(genpath('/scratch/users/rshe/QTL_modeling/'));addpath(genpath('/scratch/users/rshe/F6cross_glmnet/'));fQTLmodel_timer_lambda('$Trials','$nSites','$cycles','$nSegregants','$genomeSize','$RR','$pSites','$noise','$lambda');exit"



