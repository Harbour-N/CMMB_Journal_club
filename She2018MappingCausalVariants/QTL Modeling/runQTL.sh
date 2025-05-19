



cd $SCRATCH/QTL_modeling/

for ((nSites=1000;nSites<=40000;nSites+=1000));
do
for ((nSegregants=100;nSegregants<=4000;nSegregants+=100));
do
if [ -f 'YeastGenome-Trials_100_nSites_'$nSites'_cycles_6_nSegregants_'$nSegregants'_.mat' ]
then
   echo 'File '$nSites' '$nSegregants' exists.'
else
   echo 'File '$nSites' '$nSegregants' does not exist.'
   sbatch /scratch/users/rshe/QTL_modeling/fQTL.sbatch $nSites $nSegregants
fi
done
done



cd $SCRATCH/QTL_modeling/

for ((pSites=10;pSites<=200;pSites+=10));
do
for noise in .05 .1 .15 .2 .25 .3 .35 .4 .45 .5 .55 .6 .65 .7 .75 .8 .85 .9 .95 1;
do
if [ -f 'YeastGenome-Trials_100_pSites_'$pSites'_cycles_6_noise_'$noise'_.mat' ]
then
   echo 'File '$pSites' '$noise' exists.'
else
   echo 'File '$pSites' '$noise' does not exist.'
   sbatch /scratch/users/rshe/QTL_modeling/fQTL.sbatch $pSites $noise
fi
done
done


for lambda in -4 -3 -2 -1.8 -1.6 -1.4 -1.2 -1 -0.8 -0.6 -0.4 -0.2 0 .2 .4 .6 .8 1 1.2 1.4 1.6 1.8 2 3 4
do
   sbatch /scratch/users/rshe/QTL_modeling/fQTL_lambda.sbatch $lambda
done

for pval in 0.05 0.02 0.01 0.005 0.002 0.001 0.0005 0.0002 0.0001 0.00005 0.00002 0.00001 0.000005 0.000002 0.000001 0.0000005 0.0000002 0.0000001 0.00000001 0.000000001 0.0000000001 0.00000000001;
do
   sbatch /scratch/users/rshe/QTL_modeling/fQTL_pval.sbatch $pval
done










