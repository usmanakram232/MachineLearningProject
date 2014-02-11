%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Machine Learning Project
% Naive Bayes vs Perceptron
% Muhammad Usman Akram
% muhammadusman.akram[at]studenti.unitn.it
% 08/02/2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
%load data
load krvskp.mat % variable name data
% row major order
% numDim or  #Feature = column size - 1 [positive integer values coressponding each category]
% 1st column is class (positive integer class labels)

%randomize
[nRow, nCol] = size(data);
pIndex = randperm(nRow);
data = data(pIndex,:);
%partition
k = 10; % number of folds for cross validation
partSize = floor(nRow/k);
%testPartIndex = zeros(10,partSize);
%crossvalidate
testCVResult = zeros(10, 6);
n = 0;
for f = 1:k
    trnData = data;
    if f == k
        tstData = data(n+1:end,:);
        trnData(n+1:end,:) = [];
    else
        tstData = data(n+1:n+partSize,:);
        trnData(n+1:n+partSize,:) = [];
    end
    n = n+partSize;
    
    %yc = classifyNaiveBayes(trnData, tstData);
    yc = classifyPerceptron(trnData, tstData);
    yt = tstData(:,1);
    [Cm, tA,  ppV, Sen, F1, BAC]=getConfMtx(yt,yc);
    testCVResult(f,:) = [tA, ppV(1), ppV(2), Sen(1), Sen(2), F1];
end
%output accuracy, sensitivity, recall, F2

