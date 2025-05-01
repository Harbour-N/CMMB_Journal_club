
# loop through 53 conditions (as of 06/22/16) and run forward selection algorithm
# Condition 54 added, = normrnd(0,1,1152); Serves as a negative control (pure noise).

cd /scratch/users/rshe/F6cross_fwselection/

for i in {1..53}
do
for j in {10..20}
do
sbatch /scratch/users/rshe/F6cross_fwselection/fwselection_permutationTest090316.sbatch $i $j
echo $i $j
done
done












