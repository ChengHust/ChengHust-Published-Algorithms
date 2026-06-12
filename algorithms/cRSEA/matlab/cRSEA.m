function cRSEA(Global)
% <algorithm> <A Radial Space Division Based Evolutionary Algorithm for Many-Objective Optimization with Constraints>

% Copyright 2015-2017 Cheng He

    Population = Global.Initialization();
    Range      = inf(2,Global.M);
    while Global.NotTermination(Population)
        Range(1,:) = min([Range(1,:);Population.objs],[],1);
        Range(2,:) = max(Population(NDSort(Population.objs,1)==1).objs,[],1);
        MatingPool = TournamentSelection(2,Global.N,sum(max(0,Population.cons),2));
        Offspring  = Global.Variation(Population(MatingPool));
        Population = EnvironmentalSelection(Global,[Population,Offspring],Range,Global.N);
        if Global.evaluated>=Global.evaluation
            %Violate the constraints
            index = any(Population.cons>0,2);
            Population = Population(~index);
        end
    end
end