function gpuNet = update_net(gpuNet, resBack, lr, numDatabase, batchSize)

weightDecay = 5*1e-4 ;
numLayers = numel(gpuNet.layers) ;
for ii = 1:numLayers
    if isfield(gpuNet.layers{ii},'weights')    
        gpuNet.layers{ii}.weights{1} = gpuNet.layers{ii}.weights{1} - ...            
            lr*(resBack(ii).dzdw{1}/(batchSize * numDatabase) + weightDecay*gpuNet.layers{ii}.weights{1});    
        gpuNet.layers{ii}.weights{2} = gpuNet.layers{ii}.weights{2} - ...    
            lr*(resBack(ii).dzdw{2}/(batchSize * numDatabase) + weightDecay*gpuNet.layers{ii}.weights{2});    
    end    
end
end
