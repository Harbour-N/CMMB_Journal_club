



cd $SCRATCH/QTL_modeling/yeast/

for ((nSites=1000;nSites<=40000;nSites+=1000));
do
for ((nSegregants=100;nSegregants<=4000;nSegregants+=100));
do
if [ -f 'fQTLmodel_fineMapping-Trials_3_nSites_'$nSites'_cycles_6_nSegregants_'$nSegregants'_genomeSize_12000000_RR_3.4e-06_pSites_100_noise_0.4.mat' ]
then
   echo 'File '$nSites' '$nSegregants' exists.'
else
   echo 'File '$nSites' '$nSegregants' does not exist.'
   sbatch /scratch/users/rshe/QTL_modeling/fQTL_finemapping_090616_F6yeast.sbatch $nSites $nSegregants
fi
done
done


