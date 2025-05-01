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






