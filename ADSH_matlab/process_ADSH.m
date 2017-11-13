function result = process_ADSH(dataset, param)
%% training procedure
numDatabase = numel(param.indexRetrieval);
bit = param.bit;
gamma = param.gamma;
Sc = param.numSample;
batchSize = param.batchSize;
lr = param.lr;

XTrain = dataset.IAll(:, :, :, param.indexRetrieval);
trainLabel = dataset.LAll(param.indexRetrieval, :);
XQuery = dataset.IAll(:, :, :, param.indexQuery);
queryLabel = dataset.LAll(param.indexQuery, :);
Wtrue = queryLabel * trainLabel' > 0;

outIter = param.outIter;
maxIter = param.maxIter;

net = load('./data/imagenet-vgg-f.mat');
net = net_structure(net, bit);

V = zeros(numDatabase, bit);

loss = [];
imap = [];

train_time = 0.0;

for iter = 1: outIter
    epoch_time = tic;
    %%sampling
    Omega = randperm(numDatabase, Sc);
    sampleLabel = trainLabel(Omega, :);
    
    %%soft-constraint
    sampleS = sampleLabel * trainLabel' > 0;
    r = sum(sampleS(:)) / sum(1 - sampleS(:));
    s1 = 1;
    s0 = s1 * r;
    sampleS = sampleS*(s1 + s0) - s0;
    
    U = randn(Sc, bit);
    SQuery = XTrain(:, :, :, Omega);
    %%learning deep neural network
    for epoch = 1: maxIter
        ind = (iter - 1)*maxIter + epoch;
        index = randperm(Sc);
        for ii = 0: ceil(Sc / batchSize) - 1
            ix = index((1+ii*batchSize):min((ii+1)*batchSize, Sc));
            im = single(SQuery(:, :, :, ix));
            im_ = imresize(im, net.meta.normalization.imageSize(1: 2));
            im_ = im_ - repmat(net.meta.normalization.averageImage, 1, 1, 1, size(im_, 4));
            im_ = gpuArray(im_);
            res = vl_simplenn(net, im_);
            z = squeeze(gather(res(end).x))';
            a = tanh(z);
            bd = V(Omega, :);
            Sa = sampleS(ix, :);
            U(ix, :) = a;
            dJda = 2*(a*V' - bit*Sa) * V + 2 * gamma * (a - bd(ix, :));
            dJdz = dJda.*(1-a.^2);
            dJdoutput = gpuArray(reshape(dJdz', [1, 1, size(dJdz, 2), size(dJdz, 1)]));
            res = vl_simplenn(net, im_, dJdoutput);
            net = update_net(net, res, lr(ind), numDatabase, batchSize);
        end
    end
    H = sign(U);
    barH = zeros(numDatabase, bit);
    barH(Omega, :) = U;
    for ii = 1: bit
        V_ = V;
        V_(:, ii) = [];
        Q = -2 * bit * sampleS' * H - 2 * gamma * barH;
        q = Q(:, ii);
        H_ = U;
        h = H_(:, ii);
        H_(:, ii) = [];
        V(:, ii) = sign(-2 * V_ * H_' * h - q);
    end
    epoch_time = toc(epoch_time);
    train_time = train_time + epoch_time / 60;
    %%iteration finishes
    if strcmp(param.dataname, 'NUS-WIDE')
        fprintf('[Iteration: %3d/%3d][Train Time: %.4f (m)]\n', iter, outIter, train_time);
    else
        l = calcLoss(V, U, Omega, sampleS, gamma, bit);
        loss = [loss, l];
        fprintf('[Iteration: %3d/%3d][Loss: %.3f][Train Time: %.4f]\n', iter, outIter, l, train_time);
        if mod(iter, 10) == 0
            qB = encoding(XQuery, net, bit);
            qB = compactbit(qB > 0);
            rB = compactbit(V > 0);
            Dhamm = hammingDist(qB, rB);
            map_ = callMap(Wtrue, Dhamm);
            imap = [imap, map_];
            fprintf('[Iteration: %3d/%3d][Loss: %3.3f][MAP: %3.4f]\n', iter, outIter, l, map_);
        end
    end
end

%% Evaluation procedure
qB = encoding(XQuery, net, bit);
qB = compactbit(qB > 0);
rB = compactbit(V > 0);

if strcmp(param.dataname, 'NUS-WIDE')
    [topkmap] = calculateTopMap(rB, qB, trainLabel, queryLabel, param.topk);
    result.topkmap = topkmap;
else
    Dhamm = hammingDist(qB, rB);
    map = callMap(Wtrue, Dhamm);
    result.map = map;
    result.loss = loss;
    result.imap = imap;
end
result.qB = qB;
result.rB = rB;
end

function qB = encoding(X, net, bit)
numQuery = size(X, 4);
batchSize = 128;

qB = zeros(numQuery, bit);
for j = 0:ceil(numQuery/batchSize)-1
    ix = (1+j*batchSize):min((j+1)*batchSize, numQuery);
    im = single(X(:,:,:,ix));
    im_ = imresize(im,net.meta.normalization.imageSize(1:2));
    im_ = im_ - repmat(net.meta.normalization.averageImage,1,1,1,size(im_,4));
    im_ = gpuArray(im_);
    res = vl_simplenn(net, im_) ;
    features = squeeze(gather(res(end).x))' ;
    qB(ix,:) = features ;
end

qB = sign(qB);

end
function loss = calcLoss(Bd, Bq, Omega, S, gamma, bit)
[numQuery, numDatabase] = size(S);
l1 = norm(bit*S-Bq*Bd','fro')^2 / numDatabase;
l2 = norm(Bq - Bd(Omega,:),'fro')^2 / numQuery;
loss = l1 + gamma*l2;

end