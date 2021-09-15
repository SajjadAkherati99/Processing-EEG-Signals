clear, close all, clc

load('Normal_PVC_TrainTest')
load('Train_Features')

% Train
% Classifier 3

spreadMat = [.1,.5,.9,1.5,2] ;
NMat = [5,10,15,20,25] ;
for s = 1:5
    spread = spreadMat(s) ;
    for n = 1:5 
        Maxnumber = NMat(n) ;
        ACC = 0 ;
        % 6-fold cross-validation
        for k=1:6
            train_indices = [1:(k-1)*100,k*100+1:600] ;
            valid_indices = (k-1)*100+1:k*100 ;

            TrainX = Normalized_Train_Features(:,train_indices) ;
            ValX = Normalized_Train_Features(:,valid_indices) ;
            TrainY = Train_Label(train_indices) ;
            ValY = Train_Label(valid_indices) ;

            net = newrb(TrainX,TrainY,10^-5,spread,Maxnumber) ;
            predict_y = net(ValX);
            
            Thr = 0.5 ;
            predict_y = predict_y >= Thr ;

            ACC = ACC + length(find(predict_y==ValY)) ;
        end
        ACCMat(s,n) = ACC/600 ;
    end
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
spread = spreadMat(3) ; % Best parameter found in training step
Maxnumber = NMat(4) ; % Best parameter found in training step

TrainX = Normalized_Train_Features ;
TrainY = Train_Label ;
TestX = Normalized_Test_Features ;
TestY = Test_Label ; 

net = newrb(TrainX,TrainY,10^-5,spread,Maxnumber) ;
predict_y = net(TestX);

Thr = 0.5 ;
predict_y = predict_y >= Thr ;

ACC = length(find(predict_y==TestY))/400 ;
