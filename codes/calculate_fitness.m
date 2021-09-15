function fitness = calculate_fitness(features, index, labels, type)

if (type=="unique")
    selected_faetures = features(:, unique(index));
else
    selected_faetures = features(:, index);
end
feature1 = selected_faetures(labels==0, :);
feature2 = selected_faetures(labels==1, :);
feature1 = feature1';   feature2 = feature2';

u1 = mean(feature1, 2);     u2 = mean(feature2, 2);
N1 = length(feature1(1, :));    N2 = length(feature2(1, :));
f1 = feature1-u1;   f2 = feature2-u2;   
S1 = (f1*f1')/N1;   S2 = (f2*f2')/N2;
SW = S1+S2;
sf = selected_faetures';
u0 = mean(sf, 2);
Sb = (u1-u0)*(u1-u0)' + (u2-u0)*(u2-u0)';

fitness = trace(Sb)/trace(SW);
% if(length(unique(index)) < length(index))
%     fitness = -inf;
% end
end