%%% QTL Modeling w/ many Trials
function [sensitivity,specificity,truePosMag,falseNegMag] = fQTLmodel(Trials,nSites,cycles,nSegregants,genomeSize,RR,pSites,noise,lambda)

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

truePos = zeros(3,1);
trueNeg = zeros(3,1);
allPos = zeros(3,1);
allNeg = zeros(3,1);
for i = 1:6
    truePosMag{i} = [];
    falseNegMag{i} = [];
end

for tIndex = 1:Trials
    mBreakpoints = genomeSize*RR;
    sBreakpoints = 0;
    Coverage = 5; %Could be a function of genome size and nSegregants...budget = cov*genomeSize*nSegregants
    pMissing = poisscdf(2,Coverage); % min level of coverage needed to call a SNP
    
    nBreaks = normrnd(mBreakpoints,sBreakpoints,nSegregants,cycles);
    if cycles > 1
        nBreaks = nBreaks(:,1) + sum(nBreaks(:,2:cycles),2)/2;
    end
    nBreaks = round(nBreaks);
    nBreaks(nBreaks < 0) = 0;
    
    startingGenotype = round(rand(nSegregants,1))*2 -1;
    genotypes = ones(nSegregants,nSites);
    for i = 1:length(startingGenotype)
        genotypes(i,:) = genotypes(i,:)*startingGenotype(i);
    end
    pBreak = nBreaks/nSites;
    rBreak = rand(nSegregants,nSites); % Assumes uniform distribution of SNPs and recombination rate (not objectively true, but first order approxiation is okay)
    rMissing = rand(size(genotypes));
    rMissing = rMissing > pMissing;
    
    % Time this: would it be faster to use find(rBreak < pBreak) and then
    % multiply by some alternating flag matrix?
    for i = 1:nSegregants
        b = rBreak(i,:) < pBreak(i);
        breakpointPos = find(b);
        for j = 1:length(breakpointPos)/2
            genotypes(i,breakpointPos(2*j-1):breakpointPos(2*j)-1) = genotypes(i,breakpointPos(2*j-1):breakpointPos(2*j)-1)*-1;
        end
    end
    genotypes = genotypes.*rMissing;
     % phenotypes: norm 0, variance 1
    pMag = normrnd(0,1,pSites,1);
    pPos = unique(ceil(rand(pSites,1)*nSites));
    while length(pPos) ~= pSites
        pPos = [pPos; unique(ceil(rand(pSites-length(pPos),1)*nSites))];
    end
    phenotypes = zeros(nSegregants,1);
    for i = 1:length(phenotypes)
        phenotypes(i) = sum(genotypes(i,pPos).*pMag');
    end
    % Adjust variance and mean of phenotypes to 1 and 0.
    pMag = pMag/std(phenotypes);
    phenotypes = phenotypes-mean(phenotypes);
    phenotypes = phenotypes/std(phenotypes);
    rNoise = normrnd(0,var(phenotypes)*noise,nSegregants,1);    
    
    LOD = cell(3,1);
    for i = 1:length(LOD)
        LOD{i} = zeros(nSites,1);
    end
    for i = 1:nSites
        l = sum(rMissing(:,1));
        keep = 1:nSegregants;
        keep(rMissing(:,i) == 0) = [];
        g=genotypes(keep,i);
        p=phenotypes(keep)+rNoise(keep);
        ssres = sum((p(g==1)-mean(p(g==1))).^2) + sum((p(g==-1)-mean(p(g==-1))).^2);
        sstot = sum((p-mean(p)).^2);
        rsquare = 1-ssres/sstot;
%        r{1} = corr(genotypes(keep,i),phenotypes(keep)+rNoise(keep));
%         r{2} = corr(genotypes(:,i),phenotypes+rNoise);
%         r{3} = corr(genotypes(:,i),phenotypes);
        
        LOD{1}(i) = -(l)'.*log(1-rsquare)/(2*log(10)); % with missing genotypes + noise
%         LOD{2}(i) = -(nSegregants)'.*log(1-r{2}^2)/(2*log(10)); % noise only
%         LOD{3}(i) = -(nSegregants)'.*log(1-r{3}^2)/(2*log(10)); % perfect genotyping + phenotyping
    end
    
    % Elastic net regression
    options = glmnetSet;
    options.alpha = 0.99;
    fit = glmnet(genotypes,phenotypes,'gaussian',options);
    cvfit = cvglmnet(genotypes,phenotypes,'gaussian',options);
    lambda_2se = cvfit.lambda_min+lambda*(cvfit.lambda_1se-cvfit.lambda_min);
    lambdaIndex=sum(lambda_2se<cvfit.glmnet_fit.lambda) + 1;
    dev = cvfit.glmnet_fit.dev(lambdaIndex);
    c=glmnetCoef(fit,lambda_2se);
    cPos = find(c~=0);
    cPos(1) = [];
    cPos = cPos-1;
    c(1)=[];
    
    % Forward selection
    [b_fwselection,se,pval,inmodel,stats,nextstep,history] = stepwisefit(genotypes,phenotypes,'penter',0.0001,'display','off');
    dev_fwselection = 1-stats.SSresid/stats.SStotal;
    dof_fwselection = stats.df0;
    bPos = find(inmodel);
    
    LODpeaks = LOD{1}(pPos);
    [lodY, lodPos widths prominance] = ...
        findpeaks(LOD{1},1:nSites,'MinPeakProminence',4,...
        'MinPeakDistance',20,'MinPeakHeight',6,'MinPeakWidth',5);
    
% LOD
    [query loc] = ismember(lodPos,pPos);    
    loc(loc==0) = [];
    [query2 loc2] = ismember(1:pSites,loc);
    truePosMag{1} = [truePosMag{1};pMag(loc)];
    falseNegMag{1} = [falseNegMag{1};pMag(find(~query2))];
    truePosMag{4} = [truePosMag{4}; LOD{1}(pPos(loc))];
    falseNegMag{4} = [falseNegMag{4};LOD{1}(pPos(find(~query2)))];
    truePos(1) = truePos(1) + sum(query);
    trueNeg(1) = trueNeg(1) + nSites - pSites - (length(lodPos) - sum(ismember(lodPos,pPos)));
    allPos(1) = allPos(1) + pSites;
    allNeg(1) = allNeg(1) + nSites - pSites;
    % elastic net
    [query loc] = ismember(cPos,pPos);
    loc(loc==0) = [];
    [query2 loc2] = ismember(1:pSites,loc);
    truePosMag{2} = [truePosMag{2};pMag(loc)];
    falseNegMag{2} = [falseNegMag{2};pMag(find(~query2))];
    truePosMag{5} = [truePosMag{5}; c(pPos(loc))];
    falseNegMag{5} = [falseNegMag{5};c(pPos(find(~query2)))];
    truePos(2) = truePos(2) + sum(ismember(cPos,pPos));
    trueNeg(2) = trueNeg(2) + nSites - pSites - (length(cPos) - sum(query));
    allPos(2) = allPos(2) + pSites;
    allNeg(2) = allNeg(2) + nSites - pSites;
    % stepwise selection
    [query loc] = ismember(bPos,pPos);
    loc(loc==0) = [];
    [query2 loc2] = ismember(1:pSites,loc);
    truePosMag{3} = [truePosMag{3};pMag(loc)];
    falseNegMag{3} = [falseNegMag{3};pMag(find(~query2))];
    truePosMag{6} = [truePosMag{6}; b_fwselection(pPos(loc))];
    falseNegMag{6} = [falseNegMag{6};b_fwselection(pPos(find(~query2)))];
    truePos(3) = truePos(3) + sum(ismember(bPos,pPos));
    trueNeg(3) = trueNeg(3) + nSites - pSites - (length(bPos) - sum(query));
    allPos(3) = allPos(3) + pSites;
    allNeg(3) = allNeg(3) + nSites - pSites;
end

for i = 1:3
    sensitivity(i) = truePos(i)/allPos(i);
    specificity(i) = trueNeg(i)/allNeg(i);
end


% 
% h1 = figure(1); hold on;
% scatter(QTLsize2,LODpeaks2,'b');
% set(gca,'FontSize',30);
% title('QTL size');
% xlabel('Effect Size');
% ylabel('LOD');
% saveas(h1,'QTLsize','png');
% 
% h2 = figure(2); hold on;
% scatter(QTLsize3,d3,'b');
% set(gca,'FontSize',30);
% title('Single-nucleotide resolution');
% xlabel('Effect Size');
% ylabel('Peak calling error');
% saveas(h2,'Resolution','png');
