function [best_fitness, best_features] = genetic(features, labels, population, ...
                                                  iteration, k, trans_size, p_mu, ...
                                                    race_num, p_cross, type)

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
    
    next_pop = zeros(population, k);
    [~, next] = maxk(pop_fitness, trans_size);
    for p = 1:trans_size
        next_pop(p, :) = pop(next(p), :);
    end
    
    pop1 = zeros(population-trans_size, k);
    for p = 1:(population-trans_size)
        racers = randperm(population, race_num);
        [~, next] = maxk(pop_fitness(racers), 1);
        pop1(p, :) = pop(racers(next), :);
    end
    
    for p = 1:2:(population-trans_size)
        g1 = pop1(p, :);    g2 = pop1(p+1, :);
        g11 = zeros(1,k);   g22 = zeros(1,k);
        prob = rand(k);
        for kk = 1:k
            if (prob(kk)>0.5)
                g11(kk) = g1(kk);
                g22(kk) = g2(kk);
            else
                g11(kk) = g2(kk);
                g22(kk) = g1(kk);
            end
        end
        for m = 1:k
            for n = 1:k
                if (m~=n)
                    if(g11(m)==g11(n))
                        u = rand;
                        if (u < p_cross)
                            g11(m) = randi(numberOfFeatures);
                        end
                    end
                    if(g22(m)==g22(n))
                        u = rand;
                        if (u < p_cross)
                            g22(m) = randi(numberOfFeatures);
                        end
                    end
                end
            end
        end
        next_pop(trans_size+p,:) = g11;
        next_pop(trans_size+p+1,:) = g22;
    end
    pop = next_pop;
    
    for p = trans_size+1:population
        for kk = 1:k
            u = rand;
            if (u < p_mu)
                pop(p,kk) = randi(numberOfFeatures);
            end
        end
    end
    for p = 1:population
        pop_fitness(p) = calculate_fitness(features, pop(p,:), labels, type);
    end
end

[best_fitness, best] = maxk(pop_fitness, 1);
best_features = pop(best, :);
end