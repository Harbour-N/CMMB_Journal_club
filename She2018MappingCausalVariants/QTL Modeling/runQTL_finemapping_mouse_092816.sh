



cd $SCRATCH/QTL_modeling/mouse

for ((nSites=1000;nSites<=20000;nSites+=1000));
do
for ((nSegregants=100;nSegregants<=4000;nSegregants+=100));
do
if [ -f 'fQTLmodel_fineMapping-Trials_3_nSites_'$nSites'_cycles_12_nSegregants_'$nSegregants'_genomeSize_3179000000_RR_4.2812e-09_pSites_100_noise_0.4.mat' ]
then
   echo 'File '$nSites' '$nSegregants' exists.'
else
   echo 'File '$nSites' '$nSegregants' does not exist.'
   sbatch /scratch/users/rshe/QTL_modeling/fQTL_finemapping_092416_mouse.sbatch $nSites $nSegregants
fi
done
done


