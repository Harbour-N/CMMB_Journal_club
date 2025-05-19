function [] = fQTLmodel_timer_fineMapping_bootstrap_112016(Trials,nSites,cycles,nSegregants,genomeSize,RR,pSites,noise,loopIndex)

% cd('/Users/richardshe/Dropbox/Jarosz Lab Project/Presentations/QTL Modeling');
% Trials = 2;
% nSites = 10000;
% cycles = 6;
% nSegregants = 1000;
% genomeSize = round(10^3.445293*10^6);
% RR = 10^-0.2924298*0.01/10^6;
% pSites = 100;
% noise = 0.4;

% cd('/Volumes/Sherlock_SCRATCH/QTL_modeling/');
% Trials = 1;
% nSites = 5000;
% cycles = 6;
% nSegregants = 400;
% genomeSize = 12000000;
% RR = 3.4*10^-6;
% pSites = 100;
% noise = 0.4;

if ischar(Trials)
    Trials = str2num(Trials);
end
if ischar(nSites)
    nSites = str2num(nSites);
end
if ischar(cycles)
    cycles = str2num(cycles);
end
if ischar(nSegregants)
    nSegregants = str2num(nSegregants);
end
if ischar(genomeSize)
    genomeSize = str2num(genomeSize);
end
if ischar(RR)
    RR = str2num(RR);
end
if ischar(pSites)
    pSites = str2num(pSites);
end
if ischar(noise)
    noise = str2num(noise);
end
if ischar(loopIndex)
    loopIndex = str2num(loopIndex);
end

[sensitivity,specificity,precision,truePosMag,falseNegMag,allPks,allResolution,allSites,allPMag,allbPosAccuracy,allVariance,lod,b_fwselection,stats] = fQTLmodel_fineMapping_boostrap_112016(Trials,nSites,cycles,nSegregants,genomeSize,RR,pSites,noise,loopIndex);
% ~10 trials take about 1.5 minutes, so 100 trials would take 15 minutes, give ~1000 QTL peaks to evaluate.
% 15 minutes per phase point * 400 points in phase space = manageable.


save(['fQTLmodel_simulation_bootstrap-Trials_',num2str(Trials),'_nSites_',num2str(nSites),...
    '_cycles_',num2str(cycles),'_nSegregants_',num2str(nSegregants),'_genomeSize_',num2str(genomeSize),...
    '_RR_',num2str(RR),'_pSites_',num2str(pSites),'_noise_',num2str(noise),'_loopIndex_',num2str(loopIndex),'.mat']);
