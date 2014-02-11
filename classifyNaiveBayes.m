%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function yc = classifyNaiveBayes(trnData, tstData)
%
% Machine Learning Project
% Naive Bayes vs Perceptron
% Muhammad Usman Akram
% muhammadusman.akram[at]studenti.unitn.it
% 08/02/2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function yc = classifyNaiveBayes(trnData, tstData)
% yc = max_c_i [sumlog(P(attribute_j | c_i))] taking sum of logs of probabilities to avoid
% underflow
cv = unique(trnData(:,1)); % binary classification so 1,2
classDist = hist(trnData(:,1),2)/size(trnData,1); % class distribution
%calculating probabilities given trining set
% 2 classes, M features, K categories
% put attribute as a cell, and category x classes
pA = {};
for tr_j = 1:size(trnData,2)-1
    att = trnData(:,tr_j+1);
    uVatt = unique(att);
    ptemp = zeros(length(uVatt),3);
    for v = 1:length(uVatt)
        ptemp(v,:) = [uVatt(v), hist(trnData(att == uVatt(v), 1),2)/sum(att == uVatt(v))];
    end
    pA{tr_j} = ptemp;
end

% Calculate probability of
yc = zeros(size(tstData,1),1);
for i = 1:size(tstData,1)
    pC = [0, 0];
    for j = 1:size(tstData,2)-1
        aV = tstData(i,j+1);
        pt = pA{j};
        pc = pt(pt(:,1) == aV, 2:end);
        if size(pc,1) > 0
            pC(1) = pC(1) + log(pc(1)) + log(classDist(1));
            pC(2) = pC(2) + log(pc(2)) + log(classDist(2));
        end
    end
    pC(1) = exp(pC(1));
    pC(2) = exp(pC(2));
    if pC(1)>pC(2)
        yc(i) = cv(1);
    else
        yc(i) = cv(2);
    end
end

end
