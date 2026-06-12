function POCEA(Global)
% <algorithm> <A-G>
% Paired Offspring Generation for Constrained Large-scale Multiobjective Optimization
%--------------------------------------------------------------------------
% The copyright of the PlatEMO belongs to the BIMK Group. You are free to
% use the PlatEMO for research purposes. All publications which use this
% platform or any code in the platform should acknowledge the use of
% "PlatEMO" and reference "Ye Tian, Ran Cheng, Xingyi Zhang, and Yaochu
% Jin, PlatEMO: A MATLAB Platform for Evolutionary Multi-Objective
% Optimization, 2016".
%--------------------------------------------------------------------------

% Copyright (c) 2020-2021 Cheng He
    %% Parameter settings
	k = Global.ParameterSet(5);
	Population    = Global.Initialization();
    Lower         = min(Population.objs,[],1);
    [V0,Global.N] = UniformPoint(Global.N,Global.M);
    [Vs0,L]       = UniformPoint(floor(Global.N/k),Global.M);
    [V,Vs]        = deal(V0,Vs0);
    %% Optimization
    while Global.NotTermination(Population)
        [INDEX,THETA,DIS] = Association(Population,Vs,k);
		CV = sum(max(Population.cons,0),2);
        rf = sum(CV<1e-6)/length(Population);
        Offspring = [];
		
        %% Paired offspring generation
        for i = 1 : L
			% Subpopulation construction
            SubPop = Population(INDEX(:,i));
            theta  = THETA(INDEX(:,i));
            if mean(theta) >= pi/L/2
                [~,index] = sort(DIS);
				selected  = index(1:min(k,length(index)));
                SubPop    = [SubPop,Population(selected)];
                epsilon   = max(CV([INDEWX(:,i);selected]));
            else
                epsilon   = min(CV(INDEX(:,i)))*(1-rf)+mean(CV(INDEX(:,i)))*rf;
            end
            % Offspring generation
            if length(SubPop) < 2
                rank = [1, 1];
            else
                [~,rank]= sort(rand(k,length(SubPop)),2);
            end
            [winner,loser] = CHP(SubPop(rank(1)),SubPop(rank(2)),epsilon);
			Offspring      = [Offspring,Operator(loser,winner)];
        end 
        Lower       = min([Lower;Offspring.objs],[],1);
        Population  = RVEASelection(Lower,[Population,Offspring],V);
         if ~mod(Global.gen,ceil(0.1*Global.maxgen))
             [V,Vs] = ReferenceVectorAdaptation(Population.objs,V0,Vs0);
         end
    end
end


function [INDEX,THETA,DIS] = Association(Population,V,k)
% Associate k candidate solutions to each reference vector

    PopObj    = Population.objs - repmat(min(Population.objs,[],1),length(Population),1);
	DIS	      = sum(PopObj.^2,2);
    THETA     = acos(1 - pdist2(PopObj,V,'cosine'));
    [~,index] = sort(THETA,1);
    INDEX     = index(1:min(k,length(index)),:);
end

