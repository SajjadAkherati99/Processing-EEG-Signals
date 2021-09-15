load('Normal_PVC_TrainTest')

% Plot signals
% Normal 
tr = 3 ;
L = length(Train_Data{tr}(1,:)) ;
figure
subplot(2,1,1)
plot(Train_Data{tr}(1,:))
xlim([0,L]) ;
title(['Normal beat: Trial #',num2str(tr)])
subplot(2,1,2)
plot(Train_Data{tr}(2,:))
xlim([0,L]) ;

% PVC
tr = 1 ;
L = length(Train_Data{tr}(1,:)) ;
figure
subplot(2,1,1)
plot(Train_Data{tr}(1,:))
xlim([0,L]) ;
title(['PVC beat: Trial #',num2str(tr)])
subplot(2,1,2)
plot(Train_Data{tr}(2,:))
xlim([0,L]) ;

% Feature Extraction
Train_Features = [] ;
for i=1:600
    NewSig = Train_Data{i} ;
    [Train_Features(1,i),Train_Features(2,i)] = max(abs(NewSig(1,:))) ;
    [Train_Features(3,i),Train_Features(4,i)] = max(abs(NewSig(2,:))) ;
    Train_Features(5,i) = var(NewSig(1,:)) ;
    Train_Features(6,i) = var(NewSig(2,:)) ;
    Train_Features(7,i) = corr(NewSig(1,:)',NewSig(2,:)') ;
    
end

% Normalization
[Normalized_Train_Features,xPS] = mapminmax(Train_Features) ;

% [Normalized_Train_Features,xPS] = mapstd(Train_Features,0,1) ;

% Plot features
PVC_indices = find(Train_Label==1) ;
Normal_indices = find(Train_Label==0) ;

figure
plot3(Normalized_Train_Features(1,PVC_indices),Normalized_Train_Features(2,PVC_indices),Normalized_Train_Features(3,PVC_indices),'*r') ;
hold on
plot3(Normalized_Train_Features(1,Normal_indices),Normalized_Train_Features(2,Normal_indices),Normalized_Train_Features(3,Normal_indices),'og') ;
title('Fetures #1, #2, #3') ;


figure
plot3(Normalized_Train_Features(4,PVC_indices),Normalized_Train_Features(5,PVC_indices),Normalized_Train_Features(6,PVC_indices),'*r') ;
hold on
plot3(Normalized_Train_Features(4,Normal_indices),Normalized_Train_Features(5,Normal_indices),Normalized_Train_Features(6,Normal_indices),'og') ;
title('Fetures #4, #5, #6') ;


figure
plot3(Normalized_Train_Features(1,PVC_indices),Normalized_Train_Features(4,PVC_indices),Normalized_Train_Features(7,PVC_indices),'*r') ;
hold on
plot3(Normalized_Train_Features(1,Normal_indices),Normalized_Train_Features(4,Normal_indices),Normalized_Train_Features(7,Normal_indices),'og') ;
title('Fetures #1, #4, #7') ;

save('Train_Features','Normalized_Train_Features','xPS')
