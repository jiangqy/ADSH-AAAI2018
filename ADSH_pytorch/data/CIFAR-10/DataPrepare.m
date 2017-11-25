function DataPrepare
X = [] ;
L = [] ;
for i=1:5
    clear data labels batch_label;
    load(['./data/cifar-10-batches-mat/data_batch_' num2str(i) '.mat']);
    data = reshape(data',[32,32,3,10000]);
    data = permute(data,[2,1,3,4]);
    X = cat(4,X,data) ;
    L = cat(1,L,labels) ;
end
clear data labels;
load('./data/cifar-10-batches-mat/test_batch.mat');
data=reshape(data',[32,32,3,10000]);
data = permute(data,[2,1,3,4]);
X = cat(4,X,data) ;
L = cat(1,L,labels) ;


test_data = [];
test_L = [];
data_set = [];
dataset_L = [];
train_data = [];
train_L = [];
for label=0:9
    index = find(L==label);
    N = size(index,1) ;
    perm = randperm(N) ;
    index = index(perm);

    data = X(:,:,:,index(1:100));    
    labels = L(index(1:100));
    test_L = cat(1,test_L,labels) ;
    test_data = cat(4,test_data,data) ;  

    data = X(:,:,:,index(101:6000));    
    labels = L(index(101:6000));
    dataset_L = cat(1,dataset_L,labels) ;
    data_set = cat(4,data_set,data) ;
    
    data = X(:,:,:,index(101:600));    
    labels = L(index(101:600));
    train_L = cat(1,train_L,labels) ;
    train_data = cat(4,train_data,data) ;    
end
save('cifar-10.mat','test_data','test_L','data_set','dataset_L','train_data','train_L');
end

