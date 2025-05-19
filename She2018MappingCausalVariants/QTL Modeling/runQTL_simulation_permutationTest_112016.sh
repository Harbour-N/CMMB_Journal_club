



cd $SCRATCH/QTL_modeling/simulation_permutationTest/

nSites=12054
nSegregants=1125
for i in {1..100}
do
echo $i
if [ -f 'fQTLmodel_simulation_bootstrap-Trials_10_nSites_'$nSites'_cycles_6_nSegregants_'$nSegregants'_genomeSize_12100000_RR_3.4e-06_pSites_100_noise_0.4_loopIndex_'$i'.mat' ]
then
   echo 'File '$nSites' '$nSegregants' '$i' exists.'
else
   echo 'File '$nSites' '$nSegregants' '$i' does not exist.'
   sbatch /scratch/users/rshe/QTL_modeling/fQTL_simulation_permutationTest_112016.sbatch $nSites $nSegregants $i
fi
done
