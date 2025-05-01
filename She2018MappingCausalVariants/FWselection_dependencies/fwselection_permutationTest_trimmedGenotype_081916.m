%%%
function [] = fwselection_permutationTest_trimmedGenotype_081916(masterPath,savePath,pcutoff,p,pIndex)

if ischar(p)
    p = str2num(p);
end
if ischar(pIndex)
    pIndex = str2num(pIndex);
end
if ischar(pcutoff)
    pcutoff = str2num(pcutoff);
end
% if savePath(length(savePath)) ~= '/'
%     savePath = [savePath,'/'];
% end

cd(masterPath);
load('/scratch/users/rshe/F6cross_fwselection/filename.mat');
load('/scratch/users/rshe/F6cross_fwselection/trait.mat');
load('/scratch/users/rshe/F6cross_fwselection/permutation.mat');
load('/scratch/users/rshe/F6cross_fwselection/recursiveBootstrapCutoff2.mat');
load('/scratch/users/rshe/F6cross_fwselection/phasedGenotype080316_trimmedGenotype.mat');
load('/scratch/users/rshe/F6cross_fwselection/variantPos080316_trimmedGenotype.mat');
% Create "edge well" pseudo-genototype
col1 = ones(12,1);
col2 = [1;0;0;0;0;0;0;0;0;0;0;1];
plate = [col1;repmat(col2,6,1);col1];
edges = repmat(plate,12,1);
plateNorm = zeros(96*12,12);
for i = 1:12
    plateNorm(96*(i-1) + 1:96*i,i) = 1;
end
phasedGenotype2 = [edges,plateNorm,phasedGenotype];

% Exclude wells that don't meet coverage threshold
variantCalls = sum(abs(phasedGenotype2),1);
s = sum(abs(phasedGenotype2),2);

rowCutoff = 6000;
columnCutoff = 0;

traitRow = trait{p}(permutation{pIndex});

x=phasedGenotype2(s>rowCutoff & traitRow' ~= -1,variantCalls>columnCutoff);
% Remove edge count and plateNorm from variant calls
variantCalls(1:13)=[];
y=traitRow(s>rowCutoff & traitRow' ~= -1);
y=y';

% Normalize trait y to mean 0, variance 1
y = y - mean(y);
y = y/std(y);
outliers = find(y > 3 | y < -3);

% regress with forward selection
[b_fwselection,se,pval,inmodel,stats,nextstep,history] = stepwisefit(x,y,'penter',pcutoff,'display','off','maxiter',round(length(y)/6));
dev_fwselection = 1-stats.SSresid/stats.SStotal;
dof_fwselection = stats.df0;
bPos = find(inmodel);

stats = rmfield(stats,'xr');
stats = rmfield(stats,'covb');

% Create one unified data structure to save all relevant workspace
% variables
b.b_fwselection = b_fwselection;
b.bPos = bPos;
b.dev_fwselection = dev_fwselection;
b.stats = stats;
b.inmodel = inmodel;
b.y = y;
b.x = x;
prefix = [savePath,filename{p}(1:length(filename{p})-4),'permutationTest',num2str(pIndex)];
save([prefix,'_cutoff_',num2str(pcutoff),'_b_trimmedGenotype082416.mat'],'b','-v7.3'); 

% Weighted least squares regression;
% stdReps = error{p};
% stdReps = stdReps(s>rowCutoff & traitRow' ~= -1);
% regressionWeight = zeros(length(stdReps),1);
% for i = 1:length(stdReps)
%     regressionWeight(i) = 1/max(median(stdReps),stdReps(i));
% end
% mdl = fitlm(x(:,bPos),y,'Weights',regressionWeight);
% tempMdlResi = table2array(mdl.Residuals);
% mdlResiduals = tempMdlResi(:,1);

% Find 1D lod score for regressing each predictor one at a time
lod1D = zeros(length(x),1);
for i = 1:length(x)
    r = corr(x(:,i),y);
    lod1D(i) = -length(y)*log(1-r^2)/(2*log(10));
end

% Split variantPos into chromosome and position variables so that we can
% scan 10kb on either side of each causal variant
allPos = zeros(length(variantPos),1);
allChrom = cell(length(variantPos),1);
for i = 1:length(variantPos)
    tempVar = strsplit(variantPos{i},':');
    allChrom{i} = tempVar{1};
    allPos(i) = str2num(tempVar{2});
end

nVariants = sum(stats.PVAL>cutoff2(p));
bPos_filtered = find(-log10(stats.PVAL) > cutoff2(p));
[~,~,r_filtered] = regress(y,[ones(length(y),1),x(:,bPos_filtered)]);

startIndex = sum(bPos_filtered <= 13) + 1; % First 13 bPos rows are edge normalization and plate normalization.
for posIndex = startIndex:length(bPos_filtered)
    pos1 = bPos_filtered(posIndex);
    tempVar = strsplit(variantPos{pos1-13},':');
    chrom = tempVar{1};
    position = str2num(tempVar{2});
    % Scan all positions up to 10 kb in either direction for possible
    % causal variants that fit better by the allele swap criteria.
    lower = min(find(strcmp(chrom,allChrom) & allPos > position - 10000)) + 13;
    upper = max(find(strcmp(chrom,allChrom) & allPos < position + 10000)) + 13;
    
    for i = lower:upper
        for j = lower:upper
            [p_anova{posIndex}(i-lower+1,j-lower+1),pairwise_p{posIndex}{i-lower+1}{j-lower+1},...
                ph1{posIndex}(i-lower+1,j-lower+1),ph2{posIndex}(i-lower+1,j-lower+1)] = ...
                fineMappingLod_multiSite_anova(i,j,lower,upper,pos1,x,y,b_fwselection,r_filtered);
        end
    end
end

b.lod1D = lod1D;
b.p_anova = p_anova;
b.pairwise_p = pairwise_p;
b.ph1 = ph1;
b.ph2 = ph2;
b.cutoff2 = cutoff2(p);

save([prefix,'_cutoff_',num2str(pcutoff),'_b_trimmedGenotype082416.mat'],'b','-v7.3');


