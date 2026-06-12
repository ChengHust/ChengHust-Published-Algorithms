function Population = subCMOPSO(Global,Population,Operator,Gmax)
%Sub-optimizer in CLEA (CMOPSO)   
    %% Optimization
    for gen = 1 : Gmax
        if Operator == 1
           Offspring  = Global.Variation(Population,Global.N,@CMOPSO_operator);
        elseif Operator == 2
            Offspring = Global.Variation(Population(MatingPool),Global.N,@DE);
        else
            Offspring = Global.Variation(Population(MatingPool));
        end
        Population = CMPSOEnvironmentalSelection([Population,Offspring],Global.N);
    end
end

function Population = CMPSOEnvironmentalSelection(Population,N)
% The environmental selection of CMOPSO

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    Next = false(1,length(FrontNo));
    Next(FrontNo<MaxFNo) = true;
    PopObj = Population.objs;
    fmax   = max(PopObj(FrontNo==1,:),[],1);
    fmin   = min(PopObj(FrontNo==1,:),[],1);
    PopObj = (PopObj-repmat(fmin,size(PopObj,1),1))./repmat(fmax-fmin,size(PopObj,1),1);
    %% Select the solutions in the last front
    Last = find(FrontNo==MaxFNo);
    del  = Truncation(PopObj(Last,:),length(Last)-N+sum(Next));
    Next(Last(~del)) = true;
    % Population for next generation
    Population = Population(Next);
end

function Del = Truncation(PopObj,K)
% Select part of the solutions by truncation
    N = size(PopObj,1);
    %% Truncation
    Distance = pdist2(PopObj,PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1,N);
    while sum(Del) < K
        Remain   = find(~Del);
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end