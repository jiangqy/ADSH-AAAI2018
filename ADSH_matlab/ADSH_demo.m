function ADSH_demo()
addpath(fullfile('utils'));
dataname = 'CIFAR-10';

%% load dataset
[dataset, param] = load_data(dataname);

%% basic parameters
bits = [12, 24, 32, 48];
nb = numel(bits);

param.dataname = dataname;
param.method = 'ADSH';
param.bits = bits;
param.batchSize = 64;

%% hypper-parameters, please cross-validate the following params if you use
% these code for new datasets.
param.lr = logspace(-4, -6, param.outIter * param.maxIter);
param.outIter = 50;
param.maxIter = 3;
param.gamma = 100;
param.numSample = 1000;
if strcmp(dataname, 'NUS-WIDE')
    param.topk = 5000;
end

%% training and evaluation
for i = 1: nb
    param.bit = bits(i);
    result = process_ADSH(dataset, param);
    
    if isfield(result,'topkmap')
        fprintf('[Dataset: %s][Method: %s][Top-5000 MAP: %3.3f]', ...
            dataname, param.method, result.topkmap);
    else fprintf('[Dataset: %s][Method: %s][MAP: %3.3f]', ...
            dataname, param.method, result.map);
    end
    save(['log/ADSH_' dataname '_' int2str(param.bit) '_' datestr(now) '.mat'], 'result')
end
end

function [dataset, param] = load_data(dataname)
switch dataname
    case 'CIFAR-10'
        load ./data/CIFAR-10.mat LAll IAll param;
    case 'NUS-WIDE'
        load ./data/NUS-WIDE.mat LAll IAll param;
    case 'MS-COCO'
        load ./data/MS-COCO.mat LAll IAll param;
end
dataset.IAll = IAll;
dataset.LAll = LAll;
end

