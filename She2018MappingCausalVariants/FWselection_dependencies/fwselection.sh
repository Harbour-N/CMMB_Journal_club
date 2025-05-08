
# loop through 53 conditions (as of 06/22/16) and run forward selection algorithm
# Condition 54 added, = normrnd(0,1,1152); Serves as a negative control (pure noise).
# Conditions 55-64 added: pre-defined loci of various effect sizes added together with different levels of noise to make a phenotype vector; Serves as a positive control for sensitivity.


############ CHANGE THIS TO THE MASTER PATH ON YOUR SERVER ###################
cd /scratch/users/rshe/F6cross_fwselection/
##############################################################################

for i in {1..64}
do
############ CHANGE THIS TO THE MASTER PATH ON YOUR SERVER ###################
sbatch /scratch/users/rshe/F6cross_fwselection/fwselection.sbatch $i
##############################################################################
echo $i
done








