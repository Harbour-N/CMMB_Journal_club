



cd $SCRATCH/QTL_modeling/

for ((nSites=1000;nSites<=20000;nSites+=1000));
do
for ((nSegregants=100;nSegregants<=4000;nSegregants+=100));
do
if [ -f 'fQTLmodel-Trials_100_nSites_'$nSites'_cycles_6_nSegregants_'$nSegregants'_genomeSize_12000000_RR_5e-06_pSites_100_noise_0.4.mat' ]
then
   echo 'File '$nSites' '$nSegregants' exists.'
else
   echo 'File '$nSites' '$nSegregants' does not exist.'
   sbatch /scratch/users/rshe/QTL_modeling/fQTL_082616.sbatch $nSites $nSegregants
fi
done
done


for ((nSites=21000;nSites<=40000;nSites+=1000));
do
for ((nSegregants=100;nSegregants<=4000;nSegregants+=100));
do
if [ -f 'fQTLmodel-Trials_100_nSites_'$nSites'_cycles_6_nSegregants_'$nSegregants'_genomeSize_12000000_RR_5e-06_pSites_100_noise_0.4.mat' ]
then
   echo 'File '$nSites' '$nSegregants' exists.'
else
   echo 'File '$nSites' '$nSegregants' does not exist.'
   sbatch /scratch/users/rshe/QTL_modeling/fQTL_082616_long.sbatch $nSites $nSegregants
fi
done
done






