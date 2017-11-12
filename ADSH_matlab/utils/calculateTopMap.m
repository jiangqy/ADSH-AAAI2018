function [topkmap] = calculateTopMap(BXTrain,BXTest,XTrain_label,XTest_label,topk)
%% Calculate MAP
% Paramter:
% -Wtrue
% -Dhamm
[Ntest,~] = size(XTest_label);
ap = zeros(1,Ntest);
pre = zeros(1,Ntest);
topkap = zeros(1,Ntest);
for j = 1:Ntest
    gnd = XTest_label(j,:)*XTrain_label' > 0;
    tsum = sum(gnd);
    if tsum == 0
        continue;
    end
    ham = hammingDist(BXTest(j,:), BXTrain);
    [~,index] = sort(ham);
    gnd = gnd(index);
    count = 1:tsum;
    tindex = find(gnd == 1);
    ap(j) = mean(count./tindex);
    pre(j) = sum(gnd(1:topk))/topk;
    tgnd = gnd(1:topk);
    if sum(tgnd) == 0
        continue;
    end
    tcount = 1:sum(tgnd);
    tindex = find(tgnd == 1);
    topkap(j) = mean(tcount ./ tindex); 
end

topkmap = mean(topkap,2);
