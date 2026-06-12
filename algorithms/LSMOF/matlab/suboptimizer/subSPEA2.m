function Population = subSPEA2(Global,Population,Operator,Gmax)
%Sub-optimizer in CLEA (SPEA2)
     Fitness    = CalFitness(Population.objs);
     for i = 1 : Gmax
        MatingPool = TournamentSelection(2,Global.N,Fitness);
        if Operator == 1
            Offspring = Global.Variation(Population(MatingPool));
        else
            Offspring = Global.Variation(Population(MatingPool),Global.N,@DE);
        end
        [Population,Fitness] = SPEA2EnvironmentalSelection([Population,Offspring],Global.N);
     end
end

function Fitness = CalFitness(PopObj)
% Calculate the fitness of each solution
    N = size(PopObj,1);
    Dominate = false(N);
    for i = 1 : N-1
        for j = i+1 : N
            k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
            if k == 1
                Dominate(i,j) = true;
            elseif k == -1
                Dominate(j,i) = true;
            end
        end
    end   
    S = sum(Dominate,2);
    R = zeros(1,N);
    for i = 1 : N
        R(i) = sum(S(Dominate(:,i)));
    end
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Distance = sort(Distance,2);
    D = 1./(Distance(:,floor(sqrt(N)))+2);
    Fitness = R + D';
end

function [Population,Fitness] = SPEA2EnvironmentalSelection(Population,N)
% The environmental selection of SPEA2
    Fitness = CalFitness(Population.objs);
    Next = Fitness < 1;
    if sum(Next) < N
        [~,Rank] = sort(Fitness);
        Next(Rank(1:N)) = true;
    elseif sum(Next) > N
        Del  = Truncation(Population(Next).objs,sum(Next)-N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
    end
    Population = Population(Next);
    Fitness    = Fitness(Next);
end

function Del = Truncation(PopObj,K)
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1,size(PopObj,1));
    while sum(Del) < K
        Remain   = find(~Del);
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end