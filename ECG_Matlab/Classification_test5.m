clear, close all, clc

load('Normal_PVC_TrainTest')
load('Train_Features')

New_Train_Label = zeros(2,100) ;
New_Train_Label(1,Train_Label==1) = 1 ;
New_Train_Label(2,Train_Label==0) = 1 ;

% Train
% Classifier 4

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
            TrainY = New_Train_Label(:,train_indices) ;
            ValY = New_Train_Label(:,valid_indices) ;

            net = newrb(TrainX,TrainY,10^-5,spread,Maxnumber) ;
            predict_y = net(ValX);
            
            [maxval,mindx] = max(predict_y) ;
            p_ValY = zeros(2,100) ;
            p_ValY(1,find(mindx==1)) = 1 ;
            p_ValY(2,find(mindx==2)) = 1 ;

            ACC = ACC + length(find(p_ValY(1,:)==ValY(1,:))) ;
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

%
New_Test_Label = zeros(2,100) ;
New_Test_Label(1,Test_Label==1) = 1 ;
New_Test_Label(2,Test_Label==0) = 1 ;

% Classification
spread = spreadMat(3) ; % Best parameter found in training step
Maxnumber = NMat(5) ; % Best parameter found in training step

TrainX = Normalized_Train_Features ;
TrainY = New_Train_Label ;
TestX = Normalized_Test_Features ;
TestY = New_Test_Label ; 

net = newrb(TrainX,TrainY,10^-5,spread,Maxnumber) ;
predict_y = net(TestX);

[maxval,mindx] = max(predict_y) ;
p_TestY = zeros(2,100) ;
p_TestY(1,find(mindx==1)) = 1 ;
p_TestY(2,find(mindx==2)) = 1 ;
        

ACC = length(find(p_TestY(1,:)==TestY(1,:)))/400 ;
