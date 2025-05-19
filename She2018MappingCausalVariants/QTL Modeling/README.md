### QTL Modelling for She and Jarosz (2018)

This is the code sent by Richard She via email on 16 May 2025. 

Richard She, Daniel F. Jarosz,
Mapping Causal Variants with Single-Nucleotide Resolution Reveals Biochemical Drivers of Phenotypic Change,
Cell,
Volume 172, Issue 3,
2018,
Pages 478-490.e15,
ISSN 0092-8674,
[https://doi.org/10.1016/j.cell.2017.12.015.](https://doi.org/10.1016/j.cell.2017.12.015.)

Table of files in the folder: 

|DD|MMM|YYYY|filename|info|
|21|Mar|2016|fQTL.sbatch|run fQTLmodel_timer.m with pSites and noise from input arguments|
| 6|Sep|2016|fQTL_082616.sbatch|run fQTLmodel_timer.m with nSites and nSegregants from input arguments|
| 6|Sep|2016|fQTL_082616_long.sbatch|same at fQTL_082616.sbatch with fewer trials and higher RR (recombination rate?)|
| 4|Oct|2016|fQTL_finemapping_090616_F6yeast.sbatch||
|30|Sep|2016|fQTL_finemapping_092416_mouse.sbatch||
| 1|Oct|2016|fQTL_finemapping_092816_F1yeast.sbatch||
|29|Sep|2016|fQTL_finemapping_092816_ant.sbatch||
|30|Sep|2016|fQTL_finemapping_092816_fly.sbatch||
|21|Mar|2016|fQTL_lambda.sbatch||
|21|Mar|2016|fQTL_pval.sbatch|run fQTLmodel_timer_pval.m with pval from input argument|
|20|Nov|2016|fQTL_simulation_permutationTest_112016.sbatch||
|21|Mar|2016|fQTLmodel.m|generate genotypes, phenotype data, select markers (using LOD, glmnet and stepwise selection) and compare to actual causal markers. |
|12|Oct|2016|fQTLmodel_fineMapping_090616.m||
| 5|Oct|2016|fQTLmodel_fineMapping_100516_long.m||
|20|Nov|2016|fQTLmodel_fineMapping_boostrap_112016.m||
|21|Mar|2016|fQTLmodel_lambda.m|as fQTLmodel.m but with lambda as input argument and used in calculating lambda_2se for GLMNET|
|21|Mar|2016|fQTLmodel_pval.m|as fQTLmodel.m but with pval as input argument and used in STEPWISEFIT|
|21|Mar|2016|fQTLmodel_timer.m|call fQTLmodel.m||
| 5|Sep|2016|fQTLmodel_timer_082616.m|as FQTLmodel_timer.m except calls fQTLmodel_082616.m which doesn't actually exist!|
|12|Oct|2016|fQTLmodel_timer_fineMapping_090616.m||
|20|Nov|2016|fQTLmodel_timer_fineMapping_bootstrap_112016.m||
|21|Mar|2016|fQTLmodel_timer_lambda.m||
|21|Mar|2016|fQTLmodel_timer_pval.m|as fQTLmodel_timer.m calling fQTLmodel_pval.m (with pval argument) instead of fQTLmodel.m|
|21|Mar|2016|runQTL.sh|If data doesn't exist, run fQTL.sbatch for various settings of (nSites,nSegregants), (pSites,noise); fQTL_lambda.sbatch for set of lambda, fQTL_pval.sbatch for set of pval|
| 6|Sep|2016|runQTL_082616.sh||
| 4|Oct|2016|runQTL_finemapping_090616_F6yeast.sh||
| 1|Oct|2016|runQTL_finemapping_F1yeast_092816.sh||
|30|Sep|2016|runQTL_finemapping_ant_092816.sh||
|30|Sep|2016|runQTL_finemapping_fly_092816.sh||
|30|Sep|2016|runQTL_finemapping_mouse_092816.sh||
|20|Nov|2016|runQTL_simulation_permutationTest_112016.sh||
|11|Oct|2016|yeast|Directory of output files|
