function [] = fQTLmodel_timer_lambda(Trials,nSites,cycles,nSegregants,genomeSize,RR,pSites,noise,lambda)

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
if ischar(lambda)
    lambda = str2num(lambda);
end

[sensitivity,specificity,truePosMag,falseNegMag] = fQTLmodel_lambda(Trials,nSites,cycles,nSegregants,genomeSize,RR,pSites,noise,lambda);
% ~10 trials take about 1.5 minutes, so 100 trials would take 15 minutes, give ~1000 QTL peaks to evaluate.
% 15 minutes per phase point * 400 points in phase space = manageable.


save(['fQTLmodel-Trials_',num2str(Trials),'_nSites_',num2str(nSites),...
    '_cycles_',num2str(cycles),'_nSegregants_',num2str(nSegregants),'_genomeSize_',num2str(genomeSize),...
    '_RR_',num2str(RR),'_pSites_',num2str(pSites),'_noise_',num2str(noise),...
    '_lambda_',num2str(lambda),'.mat']);
