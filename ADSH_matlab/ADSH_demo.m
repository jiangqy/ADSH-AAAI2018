function ADSH_demo()
%% before you running this code, please change direct to matconvnet and run setup.m to setup MatConvNet.
% As this is an old version of MatConvNet (and image-net-vgg-f.mat), if you
% complie a new version, please download correpsonding pretrained
% image-net-vgg-f.mat, and maybe you need rewrite update_net function by
% yourself.
addpath(fullfile('utils'));
runtime = 1;
dataname = 'NUS-WIDE';

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
param.outIter = 50;
param.maxIter = 3;
param.gamma = 200;
param.numSample = 2000;
param.lr = logspace(-4, -6, param.outIter * param.maxIter);
% % for NUS-WIDE dataset, please use the following learning rate
% param.lr = logspace(-4.5, -6, param.outIter * param.maxIter);

if strcmp(dataname, 'NUS-WIDE')
    param.topk = 5000;
    param.lr = logspace(-4.5, -6, param.outIter * param.maxIter);
end

%% training and evaluation
for i = 1: nb
    param.bit = bits(i);
    result = process_ADSH(dataset, param);
    
    if isfield(result,'topkmap')
        fprintf('[#Bit: %3d][Dataset: %s][Method: %s][Top-5000 MAP: %.4f]\n', ...
            param.bit, dataname, param.method, result.topkmap);
    else fprintf('[#Bit: %3d][Dataset: %s][Method: %s][MAP: %.4f]\n', ...
            param.bit, dataname, param.method, result.map);
    end
    save(['log/ADSH_' dataname '_' int2str(param.bit) '_' int2str(runtime) '.mat'], 'result')
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
        param.indexRetrieval = param.indexDatabase;
end
dataset.IAll = IAll;
dataset.LAll = LAll;
end

