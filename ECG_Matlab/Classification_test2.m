clear, close all, clc

load('Normal_PVC_TrainTest')
load('Train_Features')

% Train
% Classifier 1

for N=1:10

    ACC = 0 ;
    % 6-fold cross-validation
    for k=1:6
        train_indices = [1:(k-1)*100,k*100+1:600] ;
        valid_indices = (k-1)*100+1:k*100 ;

        TrainX = Normalized_Train_Features(:,train_indices) ;
        ValX = Normalized_Train_Features(:,valid_indices) ;
        TrainY = Train_Label(train_indices) ;
        ValY = Train_Label(valid_indices) ;

        % feedforwardnet, newff, paternnet
        % patternnet(hiddenSizes,trainFcn,performFcn)
        net = patternnet(N);
        net = train(net,TrainX,TrainY);

        predict_y = net(ValX);
        Thr = 0.5 ;
        
%         p_TrainY = net(TrainX);
%         [X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,p_TrainY,1) ;
%         Thr = T(find(X==OPTROCPT(1) & Y==OPTROCPT(2))) ;
        
        predict_y = predict_y >= Thr ;

        ACC = ACC + length(find(predict_y==ValY)) ;
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

% Classification
N = 10 ; % Best parameter found in training step
TrainX = Normalized_Train_Features ;
TrainY = Train_Label ;
TestX = Normalized_Test_Features ;
TestY = Test_Label ; 

net = patternnet(N);
net = train(net,TrainX,TrainY);

predict_y = net(TestX);
Thr = 0.5 ;
predict_y = predict_y >= Thr ;

ACC = length(find(predict_y==TestY))/400 ;
