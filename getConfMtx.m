function [Cm, tA,  ppV, Sen, F1, BAC]=getConfMtx(yt,yc)
%returns the confusion matrix given true class labels yt, classified labels
%yc, Cm is the conf. mtx, tA is the total accuracy and ppV is the positive
%pridictivity value for each class
%the class labels must be in continuous natural numbers as 1,2,3...
% Lbls=1:max(yt);
yt1=zeros(size(yt));
yc1=zeros(size(yc));

cv=unique([yt;yc]);
for i=length(cv):-1:1
    %if cv(i) <= 0
        yt1(yt==cv(i))=i;
        yc1(yc==cv(i))=i;    
    %end
end

yt=yt1;yc=yc1;
    
L=max([yt;yc]);
Cm=zeros(L);
% plbls=1:L;

% indx=find(yt~=yc);
slbls=yt+L*(yc-1);
c=hist(slbls,1:L^2);
Cm(1:L^2)=c;
% True Positive   |  False Negative
% False Positive  |	 True Negative

tA=trace(Cm)/length(yt);
ppV=diag(Cm)'./sum(Cm);  % Precision = TP / (TP+FP)
Sen=diag(Cm)'./sum(Cm,2)'; % Recall =  TP / (TP+FN)
mppV = nanmean(ppV);%geomean(ppV(~isnan(ppV)));
mSen = nanmean(Sen);%geomean(Sen(~isnan(Sen)));
F1=(2*mppV*mSen)/(mppV+mSen); % mean F-Score
BAC=sum(Sen)/length(Sen); % mean Balanced accuracy


end