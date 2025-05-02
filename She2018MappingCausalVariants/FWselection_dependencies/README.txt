README

August 30th, 2017
Richard She

Instructions for performing a genotype to phenotype mapping using forward selection followed by QTN score fine mapping. As described in She and Jarosz: "Mapping causal variants with single nucleotide resolution reveals the biochemical drivers of phenotypic change."

STEP 0:
Put the entire "FWselection_dependencies" folder on your server or computer. We will refer to this folder as your "Master folder"

STEP 1:
Make a MATLAB data structure that contains your phenotype vectors.
- Create a variable "trait" in MATLAB ( trait = {}; )
- Populate each cell in "trait" with a 1x1152 phenotype vector ( trait{1} = normrnd(0,1,1,1152); %%% This example gives you a phenotype vector drawn out of a random distribution. Replace the normrnd function with your real data )
- Save the variable "trait" as a MATLAB data structure named "trait.mat" and put it in your master folder ( save('trait.mat','trait'); )

Create a MATLAB data structure that contains unique names for each phenotype vector. Make sure the length of this data structure is the same as your phenotype vector
- Create a variable "filename" in MATLAB ( filename = {}; )
- Populate each cell in "filename" with a UNIQUE name ( filename{1} = 'Random Bootstrap'; filename{2} = 'Fluconazole'; % Etc. )
- Save the variable "filename" as a MATLAB data structure named "filename" and put it in your master folder ( save('filename.mat','filename'); )

STEP 2a (Local computer): 
Run the function fwselection_trimmedGenotype.m to make a genotype to phenotype map
- Create a variable "masterPath" that contains the path to your Master Folder ( masterPath = 'PATH_TO_FOLDER'; )
- Create a variable "savePath" which will contain the output from the forward selection ( savePath = masterPath; )
- Create a variable "pcutoff" which will be the initial p-value cutoff for forward selection ( pcutoff = 0.01; )
- Create a variable "row" which will specify which trait to run the algorithm on (takes a few hours per trait)
- Change path to Master folder and add path to master folder: ( cd('PATH_TO_FOLDER'); addpath(genpath('PATH_TO_FOLDER')); )
- Run the function ( fwselection(masterPath,savePath,pcutoff,row) )

STEP 2b (Sherlock server):
Modify fwselection.sbatch and fwselection.sh to point to the location of your master folder on server.
Run fwselection.sh


ADDITIONAL INFO: 
May 2nd, 2025
Markus Owen

fwselection_trimmedGenotype.m doesn't exist. 
fwselection_permutationTest_trimmedGenotype_081916.m does exist

ORF.mat
ORFannotation.mat
classifier.mat
mutation.mat
* permutation.mat - contains "permutation", 1000 cells, each is an array of 1152 indices used as a permutation of the phenotypes (each contains 1 to 1152 in a different order). 
phasedGenotype.mat - matrix, 1152x12054, each row is a strain, each column a marker, values from {-1,0,1}, not sure why
phasedGenotype_preEmptyWells.mat
* recursiveBootstrapCutoff2.mat - "cutoff2" a simple array with 64 doubles, not sure what for. 
* trait.mat - cell array "trait" with 64 cells each with 1152 doubles (phenotypes for 1152 strains)
* variantPos.mat - 12054 cells, each with a string of variant information (so there are 12054 markers) (the info seems to be derived from "variantPos080316_trimmedGenotype.eff.vcf")
* Previous\ Traits\ and\ Filename\ MAT/filename.mat - cell array "filename" with 64 cells each with a string name for a phenotype. 
* Previous\ Traits\ and\ Filename\ MAT/trait.mat - same as trait.mat

An example of the info in variantPos.mat: 
variantPos{3003} = 'ref|NC_001137|:0169215:T:C:0:1'

NC_001137 is the reference genome for S Cerevisiae, S288c strain, Chromosome V (5)
See https://www.yeastgenome.org/strain/s288c
0169215 is the variant position
T is the reference base
C is the list of alternative alleles at that position






