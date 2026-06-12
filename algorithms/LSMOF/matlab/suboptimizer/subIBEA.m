function Population = subIBEA(Global,Population,Operator,Gmax)
%Sub-optimizer in CLEA (IBEA)   
    kappa = 0.05;
    %% Optimization
    for gen = 1 : Gmax
        MatingPool = TournamentSelection(2,Global.N,-CalFitness(Population.objs,kappa));
        if Operator == 1
            Offspring = Global.Variation(Population(MatingPool));
        else
            Offspring = Global.Variation(Population(MatingPool),Global.N,@DE);
        end
        Population = IBEAEnvironmentalSelection([Population,Offspring],Global.N,kappa);
    end
end

function Population = IBEAEnvironmentalSelection(Population,N,kappa)
% The environmental selection of IBEA
    Next = 1 : length(Population);
    [Fitness,I,C] = CalFitness(Population.objs,kappa);
    while length(Next) > N
        [~,x]   = min(Fitness(Next));
        Fitness = Fitness + exp(-I(Next(x),:)/C(Next(x))/kappa);
        Next(x) = [];
    end
    Population = Population(Next);
end

function [Fitness,I,C] = CalFitness(PopObj,kappa)
% Calculate the fitness of each solution
    N = size(PopObj,1);
    PopObj = (PopObj-repmat(min(PopObj),N,1))./(repmat(max(PopObj)-min(PopObj),N,1));
    I      = zeros(N);
    for i = 1 : N
        for j = 1 : N
            I(i,j) = max(PopObj(i,:)-PopObj(j,:));
        end
    end
    C = max(abs(I));
    Fitness = sum(-exp(-I./repmat(C,N,1)/kappa)) + 1;
end