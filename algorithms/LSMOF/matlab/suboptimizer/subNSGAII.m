function Population = subNSGAII(Global,Population,Operator,Gmax)
%Sub-optimizer in CLEA (NSGA-II)
    
        [FrontNo,~] = NDSort(Population.objs,Global.N);
        CrowdDis    = CrowdingDistance(Population.objs,FrontNo);
        for gen = 1 : Gmax
            MatingPool  = TournamentSelection(2,Global.N,FrontNo,-CrowdDis);
            if Operator == 1
                Offspring = Global.Variation(Population(MatingPool));
            else
                Offspring = Global.Variation(Population(MatingPool),Global.N,@DE);
            end
            [Population,FrontNo,CrowdDis]  = EnvironmentalSelection([Population,Offspring],Global.N);
        end
end