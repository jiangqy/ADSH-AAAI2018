function net = net_structure (net, bit)
n = numel(net.layers) - 2; 
net.layers = net.layers(1:n);

for i=1:n
    if isfield(net.layers{i}, 'weights')    
        net.layers{i}.weights{1} = gpuArray(net.layers{i}.weights{1}) ;        
        net.layers{i}.weights{2} = gpuArray(net.layers{i}.weights{2}) ;
    end   
end

net.layers{n+1}.pad = [0,0,0,0];
net.layers{n+1}.stride = [1,1];
net.layers{n+1}.type = 'conv';
net.layers{n+1}.name = 'fc8';
net.layers{n+1}.weights{1} = gpuArray(0.01*randn(1,1,4096,bit,'single'));
net.layers{n+1}.weights{2} = gpuArray(0.01*randn(1,bit,'single'));
net.layers{n+1}.opts = {};
end