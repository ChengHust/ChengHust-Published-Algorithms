function LSMOF(Global)
% <algorithm> <A-G>
%A Large-Scale Multi-Objective Optimization Framework For Faster Convergence Rate
% G1 --- 10 --- the generation of weight optimization with DE
% SubN --- 40 --- the population size of the transferred problem
% mode --- 1 --- suboptimizer is NSGA-II (1), MOEA/D-DE (2), SPEA2 (3),
% IBEA (4), else SMS-EMOA.
% operator --- 1 --- original operators (1) DE or other (2)
%--------------------------------------------------------------------------
% The copyright of the PlatEMO belongs to the BIMK Group. You are free to
% use the PlatEMO for research purposes. All publications which use this
% platform or any code in the platform should acknowledge the use of
% "PlatEMO" and reference "Ye Tian, Ran Cheng, Xingyi Zhang, and Yaochu
% Jin, PlatEMO: A MATLAB Platform for Evolutionary Multi-Objective
% Optimization, 2016".
%--------------------------------------------------------------------------

% Copyright (c) 2016-2017 Cheng He
    %% Parameter settings
    [wD,SubN,mode,Operator] =  Global.ParameterSet(10,30,6,1);
    Population  = Global.Initialization();
    G = floor(Global.evaluation*0.05/(SubN*2*wD));
    %% Optimization
    while Global.NotTermination(Population)
        if Global.evaluated < 0.5*Global.evaluation   
            Archive          = WeightOptimization(Global,G,Population,wD,SubN);
            [Population,~,~] = EnvironmentalSelection([Population,Archive],Global.N);
        else
            Population  = PFExtend(Global,Population,Operator,mode);
        end
    end
end

