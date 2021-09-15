clear, close all, clc

load('Normal_PVC_TrainTest')
load('Train_Features')

% Selecting best features based on scattering matrices

PVC_indices = find(Train_Label==1) ;
Normal_indices = find(Train_Label==0) ;

for i=1:7
    u1 = mean(Normalized_Train_Features(i,PVC_indices)) ;
    S1 = (Normalized_Train_Features(i,PVC_indices)-u1)*(Normalized_Train_Features(i,PVC_indices)-u1)' ; % =var(Normalized_Train_Features(i,PVC_indices))
    u2 = mean(Normalized_Train_Features(i,Normal_indices)) ;
    S2 = (Normalized_Train_Features(i,Normal_indices)-u2)*(Normalized_Train_Features(i,Normal_indices)-u2)' ; % =var(Normalized_Train_Features(i,Normal_indices))
    Sw = S1+S2 ;
    
    u0 = mean(Normalized_Train_Features(i,:)) ; 
    Sb = (u1-u0)^2 + (u2-u0)^2 ;
    
    J(i) = Sb/Sw ;
end
   
figure
plot3(Normalized_Train_Features(2,PVC_indices),Normalized_Train_Features(3,PVC_indices),Normalized_Train_Features(4,PVC_indices),'*r') ;
hold on
plot3(Normalized_Train_Features(2,Normal_indices),Normalized_Train_Features(3,Normal_indices),Normalized_Train_Features(4,Normal_indices),'og') ;
title('Fetures #2, #3, #4') ;

