%%% QTL Modeling w/ many Trials
function [sensitivity,specificity,precision,truePosMag,falseNegMag,allPks,allResolution,allSites,allPMag,all_bPosAccuracy,allVariance,lod,b_fwselection,stats] = fQTLmodel_fineMapping_boostrap_112016(Trials,nSites,cycles,nSegregants,genomeSize,RR,pSites,noise,loopIndex)

outputFile = fopen(['simulation_permutationTest_loopIndex_',num2str(loopIndex),'.txt'],'w');
fprintf(outputFile,['tIndex\tloopIndex\tpIndex\tvariantPos(FM)\tvariantPos(FW)\tsum_abs_x(FW)\tdev\tdof\tPVAL\tpos_fineMapping\tpos_fwselection\t',...
    'peakWidth\tpeakProminance\tb_fwselection\tLOD_pos_fineMapping\tLOD_pos_fwselection\n']);

truePos = zeros(4,1);
trueNeg = zeros(4,1);
falsePos = zeros(4,1);
allPos = zeros(4,1);
allNeg = zeros(4,1);
for i = 1:8
    truePosMag{i} = [];
    falseNegMag{i} = [];
end

allPks = cell(Trials,1);
allResolution = cell(Trials,1);
allSites = cell(Trials,1);
allPMag = cell(Trials,1);
for tIndex = 1:Trials
    mBreakpoints = genomeSize*RR;
    sBreakpoints = mBreakpoints/3; % Empirical estimation from Kruglyak data;
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
    
    %%% To account for clustering of mutations, define sites and
    %%% breakpoints independently on the genomeSize axis, count # of sites
    %%% between each breakpoint to make the genotype matrix.
    sitePositions = unique(ceil(genomeSize*rand(1,nSites)));
    while length(sitePositions) < nSites
        sitePositions = [sitePositions,unique(ceil(genomeSize*rand(1,nSites-length(sitePositions))))];
    end
    sitePositions = sort(sitePositions);
    for i = 1:length(startingGenotype);
        breakPositions = unique(ceil(genomeSize*rand(1,nBreaks(i))));
        while length(breakPositions) < nBreaks(i)
            breakPositions = [breakPositions,unique(ceil(genomeSize*rand(1,nBreaks(i)-length(breakPositions))))];
        end
        breakPositions = sort(breakPositions);
        blockPos = ones(length(breakPositions)+1,1);
        for j = 1:length(breakPositions)
            blockPos(j+1) = sum(sitePositions<breakPositions(j)) + 1;
            if mod(j,2) == 0
                genotypes(i,blockPos(j):blockPos(j+1)-1) = startingGenotype(i)*-1;
            end
        end
    end
    
    %%% Add missing genotypes based on coverage. First determine coverage
    %%% as a poisson distribution. For low coverage sites, poisscdf(1,cov) =
    %%% fraction of missing sites at that position.
    Coverage = 5; %Could be a function of genome size and nSegregants...budget = cov*genomeSize*nSegregants
    for i = 1:10
        pMissing(i) = poisspdf(i,Coverage); % min level of coverage needed to call a SNP
        covIsites = ceil(rand(ceil(pMissing(i)*nSites),1)*nSites);
        genotypes(ceil(rand(ceil(poisscdf(1,i)*nSegregants),1)*nSegregants),covIsites) = 0;
    end
            
    % phenotype magnitude: linear continum from -1 to 1
    increment = 1/pSites*2;
    pMag = [increment:increment:1,-increment:-increment:-1];
    % phenotype position: random
    pPos = unique(ceil(rand(pSites,1)*nSites));
    while length(pPos) ~= pSites
        pPos = [pPos; unique(ceil(rand(pSites-length(pPos),1)*nSites))];
    end
    pPos = unique(pPos);
    
    % Randomize pMag with respect to pPos, since both are sorted prior to
    % this step
    pMag = pMag(randperm(length(pMag)));
    phenotypes = zeros(nSegregants,1);
    for i = 1:length(phenotypes)
        phenotypes(i) = sum(genotypes(i,pPos).*pMag);
    end
    
    tempPhenotypes = zeros(nSegregants,1);
    variance = zeros(length(pMag)+1,1);
    deltaV = zeros(length(pMag),1);
    deltaV2 = zeros(length(pMag),1);
    for i = 1:length(pMag)
        tempPhenotypes = tempPhenotypes + genotypes(:,pPos(i))*pMag(i);
        variance(i+1) = var(tempPhenotypes);
        deltaV(i) = variance(i+1)-variance(i);
        tempPhenotypes2 = phenotypes - genotypes(:,pPos(i))*pMag(i);
        deltaV2(i) = var(phenotypes) - var(tempPhenotypes2);
    end
    
    % Adjust variance and mean of phenotypes to 1 and 0.
    pMag = pMag/std(phenotypes);
    phenotypes = phenotypes-mean(phenotypes);
    phenotypes = phenotypes/std(phenotypes);
    rNoise = normrnd(0,var(phenotypes)*noise,nSegregants,1);
    phenotypes = phenotypes + rNoise;
    
    %%% Save all pPos, pMag
    allSites{tIndex} = pPos;
    allPMag{tIndex} = pMag';
    allVariance{2*tIndex-1} = deltaV;
    allVariance{2*tIndex} = deltaV2;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Make RANDOM permutation of phenotype     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    rng('shuffle');
    phenotypes = phenotypes(randperm(nSegregants));
    
    for i = 1:nSites
        g=genotypes(:,i);
        r = corr(g,phenotypes);
        %        r{1} = corr(genotypes(keep,i),phenotypes(keep)+rNoise(keep));
        %         r{2} = corr(genotypes(:,i),phenotypes+rNoise);
        %         r{3} = corr(genotypes(:,i),phenotypes);
        lod(i) = -length(phenotypes).*log(1-r^2)/(2*log(10)); % with missing genotypes + noise
        %         LOD{2}(i) = -(nSegregants)'.*log(1-r{2}^2)/(2*log(10)); % noise only
        %         LOD{3}(i) = -(nSegregants)'.*log(1-r{3}^2)/(2*log(10)); % perfect genotyping + phenotyping
    end
    
    % Elastic net regression
    options = glmnetSet;
    options.alpha = 0.99;
    fit = glmnet(genotypes,phenotypes,'gaussian',options);
    cvfit = cvglmnet(genotypes,phenotypes,'gaussian',options);
    lambda_2se = cvfit.lambda_min+2*(cvfit.lambda_1se-cvfit.lambda_min);
    lambdaIndex=sum(lambda_2se<cvfit.glmnet_fit.lambda) + 1;
    dev = cvfit.glmnet_fit.dev(lambdaIndex);
    c=glmnetCoef(fit,lambda_2se);
    cPos = find(c~=0);
    if ~isempty(cPos)
        cPos(1) = [];
        cPos = cPos-1;
    end
    clear cvfit;
    clear fit;
    
    % Forward selection
    [b_fwselection,se,pval,inmodel,stats] = stepwisefit(genotypes,phenotypes,'penter',10^-5,'display','off','maxiter',round(length(phenotypes)/6));
    stats = rmfield(stats,'xr');
    stats = rmfield(stats,'covb');
    dev_fwselection = 1-stats.SSresid/stats.SStotal;
    dof_fwselection = stats.df0;
    bPos = find(inmodel);
    
    load('/scratch/users/rshe/F6cross_fwselection/recursiveBootstrapCutoff2.mat');
    % load('/Volumes/Sherlock_SCRATCH/F6cross_fwselection/recursiveBootstrapCutoff2.mat');
    nVariants = sum(stats.PVAL>mean(cutoff2));
    bPos_filtered = find(-log10(stats.PVAL) > mean(cutoff2));
    [~,~,r_filtered] = regress(phenotypes,[ones(length(phenotypes),1),genotypes(:,bPos_filtered)]);
    bPos_pkfiltered = bPos_filtered;

    toRemove = [];
    resolution = zeros(length(bPos_pkfiltered),1);
    pks = zeros(length(bPos_pkfiltered),1);
    primaryLoc = zeros(length(bPos_pkfiltered),1);
    for posIndex = 1:length(bPos_pkfiltered)
        fprintf([num2str(posIndex),'\n'])
        pos1 = bPos_filtered(posIndex);
        position = sitePositions(bPos_filtered(posIndex));
        % Scan all positions up to 10 kb in either direction for possible
        % causal variants that fit better by the allele swap criteria.
        lowerUpperInterval = round(genomeSize/nSites)*5;
        lower = min(find(sitePositions > position - lowerUpperInterval));
        upper = max(find(sitePositions < position + lowerUpperInterval));
        ph2 = zeros(upper-lower+1,upper-lower+1);
        for i = lower:upper
            for j = lower:upper
                [~,~,~,ph2(i-lower+1,j-lower+1)] = ...
                    fineMappingLod_multiSite_anova(i,j,lower,upper,pos1,genotypes,phenotypes,b_fwselection,r_filtered);
            end
        end
        %%% Find the minimum of each row of the Ph2 matrix (Ph2 is
        %%% the likelihood of the alternate hypothesis H2, and the
        %%% maximum of the Ph1 (Likelihood of H1).
        leftBound = zeros(length(ph2),1);
        rightBound = zeros(length(ph2),1);
        minPh2 = zeros(length(ph2),1);
        degeneracyCheck = zeros(length(ph2),1);
        for i = 1:length(ph2);
            if sum(genotypes(:,lower+i-1) == 1) > length(phenotypes)/4 && sum(genotypes(:,lower+i-1) == -1) > length(phenotypes)/4
                degeneracyCheck(i) = 0;
            else
                degeneracyCheck(i) = 1;
            end
        end
        
        for i = 1:length(ph2)
            temp2 = ph2(i,:);
            temp2(i) = 11; temp2(temp2 == -1) = 11; temp2(temp2 == 0) = 11; %temporary value so as not to affect min calculation, p-value is never 11 or -1;
            % If we landed in a degenerate row (i+1 and i-1 positions
            % are -1, then set to zero.
            if min(degeneracyCheck(max(1,i-1):min(length(temp2),i+1)))==1
                minPh2(i) = 0;
                leftBound(i) = 1; rightBound(i) = length(temp2);
            else
                [~,leftBound(i)] = min(temp2(1:i));
                [tempMin,rightBound(i)] = min(temp2(i:length(temp2)));
                rightBound(i) = i + rightBound(i) - 1;
                rightBound(i) = max(rightBound(i),floor((i+length(temp2))/2));
                leftBound(i) = min(leftBound(i),ceil((i+1)/2));
                
                if tempMin == 11
                    rightBound(i) = length(temp2);
                end
                temp2(i) = min(temp2); temp2(temp2 == 11) = min(temp2); % new temporary value so as not to affect min(-log10(temp2)), p-value is never 11 or -1;
                if min(temp2) ~= 11
                    minPh2(i) = min(-log10(temp2(leftBound(i):rightBound(i))));
                else
                    minPh2(i) = 0;
                end
            end
        end
        
        queryLoc = pos1 - lower + 1;
        
        %%% If the position from forward selection falls on a
        %%% degenerate column of x that does not give a good
        %%% fineMappingLod readout (usually lots of missing data or
        %%% only 1 dominant genotype at the position
        if max(ph2(queryLoc,:)) == -1
            primaryLoc(posIndex) = queryLoc;
            resolution(posIndex) = 1;
            pks(posIndex) = 0;
        else
            [pks(posIndex),primaryLoc(posIndex)] = max(minPh2(leftBound(queryLoc):rightBound(queryLoc)));
            primaryLoc(posIndex) = primaryLoc(posIndex) + leftBound(queryLoc) - 1;
            fineMappingCutoff = 0.01;
            % Find the 95% confidence interval for the causal variant
            % If all other alternative hypotheses have a likelihood <
            % 0.05, then, we have a single causal variant
            if pks(posIndex) > -log10(fineMappingCutoff);
                locs = find(minPh2(leftBound(queryLoc):rightBound(queryLoc)) > pks(posIndex)*0.7);
                locs = locs + leftBound(queryLoc) - 1;
                resolution(posIndex) = sitePositions(lower+max(locs)-1)-sitePositions(lower+min(locs)-1)+1;
                % Otherwise, find the interval (set of variants) such that
                % all other sites have p<0.05 relative to the set.
            elseif pks(posIndex) > 0
                temp2 = ph2(primaryLoc(posIndex),:);
                temp2(temp2==-1) = min(temp2(temp2>0));
                locs = [primaryLoc(posIndex),find(-log10(temp2) < -log10(fineMappingCutoff))];
                locs = unique(locs);
                temp2(min(locs):max(locs)) = min(temp2(temp2>0));
                [pks(posIndex)] = min(-log10(temp2(leftBound(primaryLoc(posIndex)):rightBound(primaryLoc(posIndex)))));
                
                % Empirical finding is that queryLoc performs better in
                % model datasets than the location found from pks.
                primaryLoc(posIndex) = queryLoc;
                resolution(posIndex) = sitePositions(lower+max(locs)-1)-sitePositions(lower+min(locs)-1)+1;
                
            else %pathological edge case
                resolution(posIndex) = sitePositions(upper)-sitePositions(lower)+1;
                primaryLoc(posIndex) = queryLoc;
                fprintf(['pks<0_fileIndex1','_posIndex',num2str(posIndex),'\n']);
            end
        end
        %%% If fineMapping position and original fwselection position do
        %%% not match
        if lower+primaryLoc(posIndex)-1 ~= pos1
            bPos_pkfiltered(posIndex) = lower+primaryLoc(posIndex)-1;
            locs = [pos1,lower+primaryLoc(posIndex)-1];
            locs = unique(locs);
            [pks(posIndex)] = minPh2(pos1-lower+1); % minPh2 at the original fw selection position
            resolution(posIndex) = sitePositions(max(locs))-sitePositions(min(locs))+1;
        end
        
        FineMappingloc = lower + primaryLoc - 1;
        fwP(posIndex) = -log10(stats.PVAL(pos1));

        %%% Print out parameters to the output File
       fprintf(outputFile,'%s\t%s\t%d\t%d\t%d\t%f\t%d\t%f\t%d\t%f\t%f\t%f\t%f\t%f\n',...
                num2str(tIndex),num2str(loopIndex),FineMappingloc,pos1,sum(abs(genotypes(:,pos1))),...
                dev_fwselection,dof_fwselection,fwP(posIndex),...
                position,resolution(posIndex),pks(posIndex),b_fwselection(pos1),...
                lod(FineMappingloc),lod(pos1));
        
        
    end    
    
    bPosAccuracy = zeros(length(bPos_pkfiltered),1);
    pPosLocIndex = zeros(length(bPos_pkfiltered),1);
    distance = cell(length(bPos_pkfiltered),1);
    for posIndex = 1:length(bPos_pkfiltered)
        distance{posIndex} = abs(sitePositions(pPos)-sitePositions(bPos_pkfiltered(posIndex)));
        temp = find(distance{posIndex} < resolution(posIndex));
        if length(temp) == 1
            bPosAccuracy(posIndex) = 1;
            pPosLocIndex(posIndex) = temp;
        end
    end
    pPosLoc = pPosLocIndex(logical(bPosAccuracy));
    [slope,bint,r] = regress(pMag(pPosLoc)',b_fwselection(bPos_pkfiltered(logical(bPosAccuracy))));
    for posIndex = 1:length(bPos_pkfiltered)
        temp = find(distance{posIndex} < resolution(posIndex));
        if length(temp) > 1
            bPosAccuracy(posIndex) = 1;
            [tempDiff,tempIndex] = min(abs(pMag(temp)-b_fwselection(bPos_pkfiltered(posIndex))));
            pPosLocIndex(posIndex) = temp(tempIndex);
        end
    end
    pPosLoc = pPosLocIndex(logical(bPosAccuracy));
    
    
    LODpeaks = lod(pPos);
    [lodY, lodPos widths prominance] = ...
        findpeaks(lod,1:nSites,'MinPeakProminence',4,...
        'MinPeakDistance',20,'MinPeakHeight',6,'MinPeakWidth',5);
    
    % LOD
    [query loc] = ismember(lodPos,pPos);
    loc(loc==0) = [];
    [query2 loc2] = ismember(1:pSites,loc);
    truePosMag{1} = [truePosMag{1};pMag(loc)'];
    falseNegMag{1} = [falseNegMag{1};pMag(find(~query2))'];
    truePosMag{4} = [truePosMag{4}; lod(pPos(loc))'];
    falseNegMag{4} = [falseNegMag{4};lod(pPos(find(~query2)))'];
    truePos(1) = truePos(1) + sum(query);
    falsePos(1) = false(1) + sum(~query);
    trueNeg(1) = trueNeg(1) + nSites - pSites - (length(lodPos) - sum(ismember(lodPos,pPos)));
    allPos(1) = allPos(1) + pSites;
    allNeg(1) = allNeg(1) + nSites - pSites;
    % elastic net
    [query loc] = ismember(cPos,pPos);
    loc(loc==0) = [];
    [query2 loc2] = ismember(1:pSites,loc);
    truePosMag{2} = [truePosMag{2};pMag(loc)'];
    falseNegMag{2} = [falseNegMag{2};pMag(find(~query2))'];
    truePosMag{5} = [truePosMag{5}; c(pPos(loc))];
    falseNegMag{5} = [falseNegMag{5};c(pPos(find(~query2)))];
    truePos(2) = truePos(2) + sum(ismember(cPos,pPos));
    falsePos(2) = falsePos(2) + sum(~ismember(cPos,pPos));
    trueNeg(2) = trueNeg(2) + nSites - pSites - (length(cPos) - sum(query));
    allPos(2) = allPos(2) + pSites;
    allNeg(2) = allNeg(2) + nSites - pSites;
    % stepwise selection
    [query loc] = ismember(bPos_filtered,pPos);
    loc(loc==0) = [];
    [query2 loc2] = ismember(1:pSites,loc);
    truePosMag{3} = [truePosMag{3};pMag(loc)'];
    falseNegMag{3} = [falseNegMag{3};pMag(find(~query2))'];
    truePosMag{6} = [truePosMag{6}; b_fwselection(bPos_filtered(find(query)))];
    falseNegMag{6} = [falseNegMag{6};b_fwselection(pPos(find(~query2)))];
    truePos(3) = truePos(3) + sum(ismember(bPos_filtered,pPos));
    falsePos(3) = falsePos(3) + sum(~ismember(bPos_filtered,pPos));
    trueNeg(3) = trueNeg(3) + nSites - pSites - (length(bPos_filtered) - sum(query));
    allPos(3) = allPos(3) + pSites;
    allNeg(3) = allNeg(3) + nSites - pSites;
    % stepwise selection _pk filtered
    loc = pPosLoc;
    [query2 loc2] = ismember(1:pSites,loc);
    truePosMag{7} = [truePosMag{7};pMag(pPosLoc)'];
    falseNegMag{7} = [falseNegMag{7};pMag(find(~query2))'];
    truePosMag{8} = [truePosMag{8}; b_fwselection(bPos_pkfiltered(logical(bPosAccuracy)))];
    falseNegMag{8} = [falseNegMag{8};b_fwselection(pPos(find(~query2)))];
    truePos(4) = truePos(4) + sum(bPosAccuracy);
    falsePos(4) = falsePos(4) + sum(~bPosAccuracy);
    trueNeg(4) = trueNeg(4) + nSites - pSites - sum(~bPosAccuracy);
    allPos(4) = allPos(4) + pSites;
    allNeg(4) = allNeg(4) + nSites - pSites;
    
    % Save pks and resolution parameters
    allPks{tIndex} = pks;
    allResolution{tIndex} = resolution;
    all_bPosAccuracy{tIndex} = bPosAccuracy;
    
    %%% Write normal parameters to outputFile
    
end

for i = 1:4
    sensitivity(i) = truePos(i)/allPos(i);
    specificity(i) = trueNeg(i)/allNeg(i);
    precision(i) = truePos(i)/(falsePos(i)+truePos(i));
end






    
    
    