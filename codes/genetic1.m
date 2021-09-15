function [best_fitness, best_features] = genetic1(features, labels, population, iteration, k, p_mu, num, type)

numberOfFeatures = length(features(1, :));
pop = zeros(population, k);
pop_fitness = zeros(population, 1);

%generate initial population
for p = 1:population
    pop(p, :) = randperm(numberOfFeatures, k);
    pop_fitness(p) = calculate_fitness(features, pop(p,:), labels, type);
end

for i = 1:iteration
    for p= 1:population
        pop(p,:) = sort(pop(p,:));
    end
    
    for p = 1:population
        popp = pop(p, :);
        delete_fitness = zeros(1,k);
        for kk = 1:k
            delete_pop = [popp(1, 1:kk-1), popp(1, kk+1:end)];
            delete_fitness(kk)= calculate_fitness(features, delete_pop, labels, type);
        end
        [~, ind] = maxk(delete_fitness,1);
        other_feature_index = setxor(randperm(numberOfFeatures,num), popp);
        add_fitness = zeros(1,length(other_feature_index));
        for p1 = 1:length(other_feature_index)
            popp(1, ind) = other_feature_index(p1);
            add_fitness(p1) = calculate_fitness(features, popp, labels, type);
        end
        [~, ind1] = maxk(add_fitness,1);
        popp(1, ind) = other_feature_index(ind1);
        pop(p,:) = popp;
    end
    prob = rand(population,k);
    for p = 1:population
        others = setxor(1:numberOfFeatures, pop(p,:));
        for kk = 1:k
            if (prob(p,kk)<p_mu)
                ind = randi(length(others));
                pop(p,kk) = others(ind); 
            end
        end
    end
end

for p = 1:population
    pop_fitness(p) = calculate_fitness(features, pop(p,:), labels, type);
end

[best_fitness, best] = maxk(pop_fitness, 1);
best_features = pop(best, :);
end