clear, close all, clc

load('Normal_PVC_TrainTest')
load('Train_Features')

New_Train_Label = zeros(2,600) ;
New_Train_Label(1,Train_Label==1) = 1 ;
New_Train_Label(2,Train_Label==0) = 1 ;

% Classifier 2

for N=1:10

    ACC = 0 ;
    % 6-fold cross-validation
    for k=1:6
        train_indices = [1:(k-1)*100,k*100+1:600] ;
        valid_indices = (k-1)*100+1:k*100 ;

        TrainX = Normalized_Train_Features(:,train_indices) ;
        ValX = Normalized_Train_Features(:,valid_indices) ;
        TrainY = New_Train_Label(:,train_indices) ;
        ValY = New_Train_Label(:,valid_indices) ;

        % feedforwardnet, newff, paternnet
        % patternnet(hiddenSizes,trainFcn,performFcn)
        net = patternnet(N);
        net = train(net,TrainX,TrainY);

        predict_y = net(ValX);

        [maxval,mindx] = max(predict_y) ;
        p_ValY = zeros(2,100) ;
        p_ValY(1,find(mindx==1)) = 1 ;
        p_ValY(2,find(mindx==2)) = 1 ;
        
        ACC = ACC + length(find(p_ValY(1,:)==ValY(1,:))) ;
        
    end

    ACCMat(N) = ACC/600 ;
end

%% Test
% Feature Exxtraction
Test_Features = [] ;
for i=1:400
    NewSig = Test_Data{i} ;
    [Test_Features(1,i),Test_Features(2,i)] = max(abs(NewSig(1,:))) ;
    [Test_Features(3,i),Test_Features(4,i)] = max(abs(NewSig(2,:))) ;
    Test_Features(5,i) = var(NewSig(1,:)) ;
    Test_Features(6,i) = var(NewSig(2,:)) ;
    Test_Features(7,i) = corr(NewSig(1,:)',NewSig(2,:)') ;
    
end

% Normalization
Normalized_Test_Features = mapminmax('apply',Test_Features,xPS) ;

%
New_Test_Label = zeros(2,100) ;
New_Test_Label(1,Test_Label==1) = 1 ;
New_Test_Label(2,Test_Label==0) = 1 ;

% Classification
N = 10 ; % Best parameter found in training step
TrainX = Normalized_Train_Features ;
TrainY = New_Train_Label ;
TestX = Normalized_Test_Features ;
TestY = New_Test_Label ; 

net = patternnet(N);
net = train(net,TrainX,TrainY);

predict_y = net(TestX);
[maxval,mindx] = max(predict_y) ;
p_TestY = zeros(2,100) ;
p_TestY(1,find(mindx==1)) = 1 ;
p_TestY(2,find(mindx==2)) = 1 ;
        

ACC = length(find(p_TestY(1,:)==TestY(1,:)))/400 ;