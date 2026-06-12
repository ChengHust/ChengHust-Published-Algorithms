function Population = subSMPSO(Global,Population,Operator,Gmax)
%Sub-optimizer in SMPSO  
    %% Optimization
	Pbest            = Population;
    [Gbest,CrowdDis] = UpdateGbest(Population,Global.N);
    for gen = 1 : Gmax
        Population  = Global.Variation([Population,Pbest,Gbest(TournamentSelection(2,Global.N,-CrowdDis))],Global.N,@SMPSO_operator);
        [Gbest,CrowdDis] = UpdateGbest([Gbest,Population],Global.N);
        Pbest            = UpdatePbest(Pbest,Population);
        drawnow()
    end
end
