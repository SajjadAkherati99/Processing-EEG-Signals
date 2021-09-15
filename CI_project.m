%%  load and plot datas
clc
clear

data = load('CI_Project_data.mat');
test_data = data.TestData;
train_data = data.TrainData;
train_label = data.TrainLabel;
clear data

%% extraxt time and frequency domain features:

features_vec_time = zeros(165, 675);
for i = 1:165
    A = train_data(:, :, i);
    A = A';
    A = cov(A);
    A = triu(A);
    for j = 465:675
        if (j <= 465)
            features_vec_time(i, 1:465) = A(A~=0);
        elseif ((465<j) && (j<=495))
            ind = j-465;
            y = train_data(ind, :, i);
            ydot = diff(y, 1);
            ydotdot = diff(ydot, 1);
            s = var(y);
            sdot = var(ydot);
            sdotdot = var(ydotdot);
            features_vec_time(i, j) = (s*sdotdot)/(sdot^2);
        elseif ((495<j) && (j<=555))
            if (mod((j-495), 2) == 1)
                ind = ceil((j-495)/2);
                A = train_data(ind, :, i);
                features_vec_time(i, j) = length(A(A < 0))/256;
                features_vec_time(i, j+1) = length(A(0 <= A))/256;
            end
        else
            if (mod((j-555), 4) == 1)
                ind = ceil((j-555)/4);
                A = train_data(ind, :, i);
                [max_data, max_ind] = maxk(A, 1);
                [min_data, min_ind] = mink(A, 1);
                features_vec_time(i, j) = max_data;
                features_vec_time(i, j+1) = max_ind;
                features_vec_time(i, j+2) = min_data;
                features_vec_time(i, j+3) = min_ind;
            end
        end
    end
end

features_vec_frequency = zeros(165, 300);
for i = 1:165
    for j = 1:30
        for k = 1:10
            ind = 10*(j-1) + k;
            signal = train_data(j, :, i);
            fs = 256;
            t = 0: 1/fs: 1-1/fs;
            energy = zeros(1,7);
            SIGNAL = abs(fftshift(fft(signal)));
            power = SIGNAL.^2;
            energy(1) = sum(SIGNAL(131:136)) + 0.5*(SIGNAL(130)+SIGNAL(137));
            energy(2) = sum(SIGNAL(138:143)) + 0.5*(SIGNAL(137)+SIGNAL(144));
            energy(3) = sum(SIGNAL(145:150)) + 0.5*(SIGNAL(144)+SIGNAL(151));
            energy(4) = sum(SIGNAL(152:157)) + 0.5*(SIGNAL(151)+SIGNAL(158));
            energy(5) = sum(SIGNAL(159:164)) + 0.5*(SIGNAL(158)+SIGNAL(165));
            energy(6) = sum(SIGNAL(166:171)) + 0.5*(SIGNAL(165)+SIGNAL(172));
            energy(7) = sum(SIGNAL(173:178)) + 0.5*(SIGNAL(172)+SIGNAL(179));
            if (k==1)
                psdest = psd(spectrum.periodogram, signal, 'Fs', 256, 'NFFT',length(signal));
                features_vec_frequency(i, ind) = max(psdest.Data);
            elseif (k==2)
                features_vec_frequency(i, ind) = meanfreq(signal, fs);
            elseif (k==3)
                features_vec_frequency(i, ind) = medfreq(signal, fs);
            else
                features_vec_frequency(i, ind) = energy(k-3)/sum(energy);
            end
        end
    end
end


%% extract effective features:
%normalize features:

clear A energy fs i j ind k max_data max_ind min_data min_ind power psdest s sdot sdotdot
clear signal SIGNAL t y ydot ydotdot

max_time = zeros(1,675);
max_frequency = zeros(1,300);

for i=1:675
    max_time(i) = max(features_vec_time(:, i));
    features_vec_time(:, i) = features_vec_time(:, i)/ max_time(i);
end

for i=1:300
    max_frequency(i) = max(features_vec_frequency(:, i));
    features_vec_frequency(:, i) = features_vec_frequency(:, i)/ max_frequency(i);
end

% caculate fisher:
fisher_time = zeros(1, 675);
fisher_frequency = zeros(1, 300);

label1 = find(train_label==1);
label0 = find(train_label==0);

for i = 1:675
    u0 = mean(features_vec_time(:, i));
    u1 = mean(features_vec_time(label0, i));
    u2 = mean(features_vec_time(label1, i));
    s1 = var(features_vec_time(label0, i));
    s2 = var(features_vec_time(label1, i));
    fisher_time(i) = ((u0-u1)^2 + (u0-u2)^2)/(s1+s2);
end

for i = 1:300
    u0 = mean(features_vec_frequency(:, i));
    u1 = mean(features_vec_frequency(label0, i));
    u2 = mean(features_vec_frequency(label1, i));
    s1 = var(features_vec_frequency(label0, i));
    s2 = var(features_vec_frequency(label1, i));
    fisher_frequency(i) = ((u0-u1)^2 + (u0-u2)^2)/(s1+s2);
end

[best_time_value, best_time_index] = maxk(fisher_time, 10);
[best_frequency_value, best_frequency_index] = maxk(fisher_frequency, 20);

best_features_time = features_vec_time(:, best_time_index);
best_features_frequency = features_vec_frequency(:, best_frequency_index);

best_features = [best_features_time, best_features_frequency];


clear best_features_frequency best_features_time best_frequency_value best_time_value
clear fisher_frequency fisher_time i label0 label1 s1 s2 u0 u1 u2

%% running k_fold by diffrent values for layers and their number of neurons

% ACCMat = zeros(20, 10);
% for N1 = 1:20
%     for N2 = 11:20
%         ACC = 0 ;
%         % 6-fold cross-validation
%         for k=1:5
%             train_indices = [1:(k-1)*33,k*33+1:165] ;
%             valid_indices = (k-1)*33+1:k*33 ;
%             
%             TrainX = best_features(train_indices, :) ;
%             ValX = best_features(valid_indices, :) ;
%             TrainY = train_label(train_indices) ;
%             ValY = train_label(valid_indices) ;
%             TrainX = TrainX';
%             ValX = ValX';
%             
%             % feedforwardnet, newff, paternnet
%             % patternnet(hiddenSizes,trainFcn,performFcn)
%             net = patternnet([N1, N2]);
%             net = train(net,TrainX,TrainY);
%             
%             predict_y = net(ValX);
%             
%             p_TrainY = net(TrainX);
%             [X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,p_TrainY,1) ;
%             Thr = T(find(X==OPTROCPT(1) & Y==OPTROCPT(2))) ;
%             
%             predict_y = predict_y >= Thr ;
%             
%             ACC = ACC + length(find(predict_y==ValY)) ;
%         end
%         
%         ACCMat(N1, N2-10) = ACC/165 ;
%     end
% end

%% experiment diffrent methods:

% method = ["trainlm", "trainbr", "trainbfg", "trainrp", "trainscg"];
% method =  [method, "traincgb", "traincgf", "traincgp", "trainoss"];
% method =  [method, "traingdx", "traingdm", "traingd"];
% ACCMat = zeros(1, 12);
% 
% % 5-fold cross-validation
% for m = 1:12
%     ACC = 0 ;
%     for k=1:5
%         train_indices = [1:(k-1)*33,k*33+1:165] ;
%         valid_indices = (k-1)*33+1:k*33 ;
%         
%         TrainX = best_features(train_indices, :) ;
%         ValX = best_features(valid_indices, :) ;
%         TrainY = train_label(train_indices) ;
%         ValY = train_label(valid_indices) ;
%         TrainX = TrainX';
%         ValX = ValX';
%         
%         % feedforwardnet, newff, paternnet
%         % patternnet(hiddenSizes,trainFcn,performFcn)
%         net = patternnet([17, 14], method(m));
%         net = train(net,TrainX,TrainY);
%         
%         predict_y = net(ValX);
%         
%         p_TrainY = net(TrainX);
%         [X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,p_TrainY,1) ;
%         Thr = T((X==OPTROCPT(1)) & (Y==OPTROCPT(2)));
%         
%         predict_y = predict_y >= Thr ;
%         ACC = ACC + length(find(predict_y==ValY)) ;
%     end
%     ACCMat(m) = ACC / 165;
% end

%% best method is "trainoss" and N1=17, N2=14; learn the best network:

ACC = 0 ;
best_fitness = 0;
best_net = 0;
for i = 1:10
    best_fitness_k = 0;
    best_net_k = 0;
    for k=1:5
        train_indices = [1:(k-1)*33,k*33+1:165] ;
        valid_indices = (k-1)*33+1:k*33 ;
        
        TrainX = best_features(train_indices, :) ;
        ValX = best_features(valid_indices, :) ;
        TrainY = train_label(train_indices) ;
        ValY = train_label(valid_indices) ;
        TrainX = TrainX';
        ValX = ValX';
        
        % feedforwardnet, newff, paternnet
        % patternnet(hiddenSizes,trainFcn,performFcn)
        net = patternnet([17, 14], "trainoss");
        net = train(net,TrainX,TrainY);
        
        predict_y = net(ValX);
        
        p_TrainY = net(TrainX);
        [X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,p_TrainY,1) ;
        Thr = T((X==OPTROCPT(1)) & (Y==OPTROCPT(2)));
        
        predict_y = predict_y >= Thr ;
        ACC_k = length(find(predict_y==ValY));
        ACC = ACC + ACC_k;
        if (ACC_k > best_fitness_k)
            best_net_k = net;
            best_fitness_k = ACC_k;
        end
    end
    ACC = ACC/165;
    if (ACC > best_fitness)
        best_net = net;
        best_fitness = ACC;
    end
end
best_fitness_MLP_phase1 = best_fitness;
best_net_MLP_phase1 = best_net;

clear ACC ACC_k AUC best_fitness_k best_net_k i k net OPTROCPT p_TrainY predict_y T
clear train_indices TrainX TrainY valid_indices ValX ValY X Y best_net best_fitness


%% extract features of test data:

% extract time domain features:
features_vec_time_test = zeros(45, 675);
for i = 1:45
    A = test_data(:, :, i);
    A = A';
    A = cov(A);
    A = triu(A);
    for j = 465:675
        if (j <= 465)
            features_vec_time_test(i, 1:465) = A(A~=0);
        elseif ((465<j) && (j<=495))
            ind = j-465;
            y = train_data(ind, :, i);
            ydot = diff(y, 1);
            ydotdot = diff(ydot, 1);
            s = var(y);
            sdot = var(ydot);
            sdotdot = var(ydotdot);
            features_vec_time_test(i, j) = (s*sdotdot)/(sdot^2);
        elseif ((495<j) && (j<=555))
            if (mod((j-495), 2) == 1)
                ind = ceil((j-495)/2);
                A = test_data(ind, :, i);
                features_vec_time_test(i, j) = length(A(A < 0))/256;
                features_vec_time_test(i, j+1) = length(A(0 <= A))/256;
            end
        else
            if (mod((j-555), 4) == 1)
                ind = ceil((j-555)/4);
                A = test_data(ind, :, i);
                [max_data, max_ind] = maxk(A, 1);
                [min_data, min_ind] = mink(A, 1);
                features_vec_time_test(i, j) = max_data;
                features_vec_time_test(i, j+1) = max_ind;
                features_vec_time_test(i, j+2) = min_data;
                features_vec_time_test(i, j+3) = min_ind;
            end
        end
    end
end

% extract frequency domain features:
features_vec_frequency_test = zeros(45, 300);
for i = 1:45
    for j = 1:30
        for k = 1:10
            ind = 10*(j-1) + k;
            signal = test_data(j, :, i);
            fs = 256;
            t = 0: 1/fs: 1-1/fs;
            energy = zeros(1,7);
            SIGNAL = abs(fftshift(fft(signal)));
            power = SIGNAL.^2;
            energy(1) = sum(SIGNAL(131:136)) + 0.5*(SIGNAL(130)+SIGNAL(137));
            energy(2) = sum(SIGNAL(138:143)) + 0.5*(SIGNAL(137)+SIGNAL(144));
            energy(3) = sum(SIGNAL(145:150)) + 0.5*(SIGNAL(144)+SIGNAL(151));
            energy(4) = sum(SIGNAL(152:157)) + 0.5*(SIGNAL(151)+SIGNAL(158));
            energy(5) = sum(SIGNAL(159:164)) + 0.5*(SIGNAL(158)+SIGNAL(165));
            energy(6) = sum(SIGNAL(166:171)) + 0.5*(SIGNAL(165)+SIGNAL(172));
            energy(7) = sum(SIGNAL(173:178)) + 0.5*(SIGNAL(172)+SIGNAL(179));
            if (k==1)
                psdest = psd(spectrum.periodogram, signal, 'Fs', 256, 'NFFT',length(signal));
                features_vec_frequency_test(i, ind) = max(psdest.Data);
            elseif (k==2)
                features_vec_frequency_test(i, ind) = meanfreq(signal, fs);
            elseif (k==3)
                features_vec_frequency_test(i, ind) = medfreq(signal, fs);
            else
                features_vec_frequency_test(i, ind) = energy(k-3)/sum(energy);
            end
        end
    end
end

clear A energy fs i j ind k max_data max_ind min_data min_ind power psdest s sdot sdotdot
clear signal SIGNAL t y ydot ydotdot

%% normalize features of test data and extract effective features:

for i=1:675
    features_vec_time_test(:, i) = features_vec_time_test(:, i)/ max_time(i);
end

for i=1:300
    features_vec_frequency_test(:, i) = features_vec_frequency_test(:, i)/ max_frequency(i);
end

best_features_time = features_vec_time_test(:, best_time_index);
best_features_frequency = features_vec_frequency_test(:, best_frequency_index);

best_features_test = [best_features_time, best_features_frequency];
best_features_test = best_features_test';

clear i best_features_time best_features_frequency
clear best_frequency_value best_time_value
clear best_frequency_index best_time_index max_frequency max_time

%% predict test labels
clc
test_labels_MLP_phase1 = best_net_MLP_phase1(best_features_test);
test_labels_MLP_phase1 = test_labels_MLP_phase1 >= Thr;
save('test_labels_MLP_phase1')
save('best_net_MLP_phase1')

%% RBF with n in range 5-20 and r in range 0.5:2 with steps of 0.1:

% r = 1:30; r = r/10;
% ACCMat = zeros(30, 15);
% for i = 5:20
%     spread = r(i);
%     for j = 3:15
%         numberOfNourouns = j;
%         ACC = 0 ;
%         for k = 1:5
%             train_indices = [1:(k-1)*33,k*33+1:165] ;
%             valid_indices = (k-1)*33+1:k*33 ;
%             
%             TrainX = best_features(train_indices, :) ;
%             ValX = best_features(valid_indices, :) ;
%             TrainY = train_label(train_indices) ;
%             ValY = train_label(valid_indices) ;
%             TrainX = TrainX';
%             ValX = ValX';
%             
%             net = newrb(TrainX, TrainY, 0, spread, numberOfNourouns);
%             
%             predict_y = net(ValX);
%             
%             p_TrainY = net(TrainX);
%             [X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,p_TrainY,1) ;
%             Thr = T((X==OPTROCPT(1)) & (Y==OPTROCPT(2)));
%             
%             predict_y = predict_y >= Thr ;
%             ACC_k = length(find(predict_y==ValY));
%             ACC = ACC + ACC_k;
%         end
%         ACCMat(i, j) = ACC/165;
%         clc
%     end
% end

%% best RBF is with r=1.3 and n=3

ACC = 0 ;

for k = 1:5
    train_indices = [1:(k-1)*33,k*33+1:165] ;
    valid_indices = (k-1)*33+1:k*33 ;
    
    TrainX = best_features(train_indices, :) ;
    ValX = best_features(valid_indices, :) ;
    TrainY = train_label(train_indices) ;
    ValY = train_label(valid_indices) ;
    TrainX = TrainX';
    ValX = ValX';
    
    net = newrb(TrainX, TrainY, 0, 1.3, 3);
    
    predict_y = net(ValX);
    
    p_TrainY = net(TrainX);
    [X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,p_TrainY,1) ;
    Thr = T((X==OPTROCPT(1)) & (Y==OPTROCPT(2)));
    
    predict_y = predict_y >= Thr ;
    ACC_k = length(find(predict_y==ValY));
    ACC = ACC + ACC_k;
end
ACC = ACC/165;
best_net_RBF_phase1 = net;
best_fitnes_RBF_phase1 = ACC;
clear ACC_k ACCMat AUC i ind j k numberOfNourouns OPTROCPT p_TrainY predict_y r spread
clear T train_indices TrainX TrainY valid_indices ValX ValY X x Y ACC net
clc

%% predict test labels

test_labels_RBF_phase1 = best_net_RBF_phase1(best_features_test);
test_labels_RBF_phase1 = test_labels_RBF_phase1 >= Thr;
save('test_labels_RBF_phase1')
save('best_net_RBF_phase1')

%% extract features base of genetic algorithm

clc
features = [features_vec_time, features_vec_frequency];

tic
[max_trace, best_index] = genetic(features, train_label, 1000, ...
    100, 50, 80, 0.1, 40, 0.25, "not_unique");
toc

% tic
% [max_trace, best_index] = genetic1(features, train_label, 100, 30, 44, 0, 50, "not_unique");
% toc

best_index = unique(best_index);
best_features = features(:, best_index);

%%
clc
for i = 1:5
    if (length(best_features(1, :))>=3)
        x = randperm(length(best_features(1, :)), 3);
        figure
        scatter3(best_features(train_label==0, x(1)), best_features(train_label==0, x(2)),...
            best_features(train_label==0, x(3)),'ro');
        hold on
        scatter3(best_features(train_label==1, x(1)), best_features(train_label==1, x(2)),...
            best_features(train_label==1, x(3)), 'bo');
    else
        plot(best_features(train_label==0, 1), best_features(train_label==0, 2), 'ro')
        hold on
        plot(best_features(train_label==1, 1), best_features(train_label==1, 2), 'bo')
    end
end

%% RBF with n in range 10-30 and r in range 1:3 with steps of 0.2:

tic
r = 1:35; r = r/10;
ACCMat = zeros(35, 35);
for i = 10:2:30
    spread = r(i);
    for j = 10:2:30
        numberOfNourouns = j;
        ACC = 0 ;
        for k = 1:5
            train_indices = [1:(k-1)*33,k*33+1:165] ;
            valid_indices = (k-1)*33+1:k*33 ;
            
            TrainX = best_features(train_indices, :) ;
            ValX = best_features(valid_indices, :) ;
            TrainY = train_label(train_indices) ;
            ValY = train_label(valid_indices) ;
            TrainX = TrainX';
            ValX = ValX';
            
            net = newrb(TrainX, TrainY, 0, spread, numberOfNourouns);
            
            predict_y = net(ValX);
            
            p_TrainY = net(TrainX);
            [X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,p_TrainY,1) ;
            Thr = T((X==OPTROCPT(1)) & (Y==OPTROCPT(2)));
            
            predict_y = predict_y >= Thr ;
            ACC_k = length(find(predict_y==ValY));
            ACC = ACC + ACC_k;
        end
        ACCMat(i, j) = ACC/165;
        clc
    end
end
a=reshape(ACCMat, [], 1);
[~,ind] = maxk(a,1);
n = ceil(ind/35);
r10 = ind-(35*(n-1));
r = r10/10;
toc
clear a ACC ACC_k ACCMat AUC i j k net numberOfNourouns OPTROCPT p_TrainY predict_y
clear r10 spread T Thr TrainX TrainY valid_indices ValX ValY X Y

%% extract best features in test vector:

test_features = [features_vec_time_test, features_vec_frequency_test];

best_features_test = test_features(:, best_index);
best_features_test = best_features_test';

%% run best RBF

ACC = 0 ;
for k = 1:5
    train_indices = [1:(k-1)*33,k*33+1:165] ;
    valid_indices = (k-1)*33+1:k*33 ;
    
    TrainX = best_features(train_indices, :) ;
    ValX = best_features(valid_indices, :) ;
    TrainY = train_label(train_indices) ;
    ValY = train_label(valid_indices) ;
    TrainX = TrainX';
    ValX = ValX';
    
    net = newrb(TrainX, TrainY, 0, r, n);
    
    predict_y = net(ValX);
    
    p_TrainY = net(TrainX);
    [X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,p_TrainY,1) ;
    Thr = T((X==OPTROCPT(1)) & (Y==OPTROCPT(2)));
    
    predict_y = predict_y >= Thr ;
    ACC_k = length(find(predict_y==ValY));
    ACC = ACC + ACC_k;
end
clc
best_fitness_RBF_phase2 = ACC/165;
best_net_RBF_phase2 = net;

clear ACC ACC_k AUC ind k net OPTROCPT p_TrainY predict_y T train_indices
clear TrainX TrainY valid_indices ValX ValY X Y

%% predict test labels RBF

test_labels_RBF_phase2 = best_net_RBF_phase2(best_features_test);
test_labels_RBF_phase2 = test_labels_RBF_phase2 >= Thr;
save('test_labels_RBF_phase2')
save('best_net_RBF_phase2')

%% run best MLP

ACC = 0 ;
best_fitness = 0;
best_net = 0;
for i = 1:10
    best_fitness_k = 0;
    best_net_k = 0;
    for k=1:5
        train_indices = [1:(k-1)*33,k*33+1:165] ;
        valid_indices = (k-1)*33+1:k*33 ;
        
        TrainX = best_features(train_indices, :) ;
        ValX = best_features(valid_indices, :) ;
        TrainY = train_label(train_indices) ;
        ValY = train_label(valid_indices) ;
        TrainX = TrainX';
        ValX = ValX';
        
        % feedforwardnet, newff, paternnet
        % patternnet(hiddenSizes,trainFcn,performFcn)
        net = patternnet([17, 14], "trainoss");
        net = train(net,TrainX,TrainY);
        
        predict_y = net(ValX);
        
        p_TrainY = net(TrainX);
        [X,Y,T,AUC,OPTROCPT] = perfcurve(TrainY,p_TrainY,1) ;
        Thr = T((X==OPTROCPT(1)) & (Y==OPTROCPT(2)));
        
        predict_y = predict_y >= Thr ;
        ACC_k = length(find(predict_y==ValY));
        ACC = ACC + ACC_k;
        if (ACC_k > best_fitness_k)
            best_net_k = net;
            best_fitness_k = ACC_k;
        end
    end
    ACC = ACC/165;
    if (ACC > best_fitness)
        best_net = net;
        best_fitness = ACC;
    end
end
best_fitness_MLP_phase2 = best_fitness;
best_net_MLP_phase2 = best_net;

clear ACC ACC_k AUC best_fitness_k best_net_k i k net OPTROCPT p_TrainY predict_y T
clear train_indices TrainX TrainY valid_indices ValX ValY X Y best_net best_fitness


%% predict test labels MLP
clc
test_labels_MLP_phase2 = best_net_MLP_phase2(best_features_test);
test_labels_MLP_phase2 = test_labels_MLP_phase2 >= Thr;
save('test_labels_MLP_phase2')
save('best_net_MLP_phase2')
