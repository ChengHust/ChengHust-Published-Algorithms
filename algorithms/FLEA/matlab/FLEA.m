classdef FLEA < ALGORITHM
% <multi/many> <real> <large/none>
% Fast sampling based evolutionary algorithm

%------------------------------- Reference --------------------------------
% L. Li, C. He, R. Cheng, H. Li, L. Pan, and Y. Jin, A fast sampling based
% evolutionary algorithm for million-dimensional multiobjective
% optimization, Swarm and Evolutionary Computation, 2022, 75: 101181.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He & Lianghao Li

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            Population = Problem.Initialization(); 
            RN         = ceil(sqrt(Problem.N));	% Number of reference solutions
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                FrontNo          = NDSort(Population.objs,Population.cons,inf);
                RefIndex         = RefSelection(Problem,Population,RN,(Problem.FE/Problem.maxFE)^2);	% APD_Based_Selection
                [id_obj,id_dec]  = Neighborhood_Association(Population,RefIndex); 
                cm               = 0.1*floor(10*(Problem.FE/Problem.maxFE));	% Ratio of convergence population
                dirV_obj         = Direction_Calculation(Problem,Population,RefIndex,id_obj,FrontNo);
                dirV_dec         = Direction_Calculation(Problem,Population,RefIndex,id_dec,FrontNo);
                Offspring        = Reproduction(Problem,Population(RefIndex).decs,cm,dirV_obj,dirV_dec);
                [Population,~,~] = EnvironmentalSelection([Population,Offspring],Problem.N);
            end
        end
    end
end

function RefIndex = RefSelection(Problem,Population,RN,theta)
% The environmental selection of RVEA
    [V,~] = UniformPoint(RN,Problem.M);
    V(1:RN,:) = V.*repmat(max(Population.objs,[],1)-min(Population.objs,[],1)+eps,size(V,1),1);
    PopObj = Population.objs;
    [N,M]  = size(PopObj);
    NV     = size(V,1);
    PopObj = PopObj - repmat(min(PopObj,[],1),N,1);
    cosine = 1 - pdist2(V,V,'cosine');
    cosine(logical(eye(length(cosine)))) = 0;
    gamma  = min(acos(cosine),[],2);

    %% Associate each solution to a reference vector
    Angle  = acos(1-pdist2(PopObj,V,'cosine'));
    Next   = zeros(1,NV);
    for i  = 1 : NV 
        APD = (1+M*theta*Angle(:,i)/gamma(i)).*sqrt(sum(PopObj.^2,2));
        [~,Next(i)] = min(APD);
        Angle(Next(i),:) = inf;
    end
    % Population for next generation
    [~,order1]  = sort(Population(Next).objs); 
    RefIndex    = Next(order1(:,1));
end

function [OCid,DCid] = Neighborhood_Association(Population,RefIndex)
    % Neighborhood in objective space
    PopDec   = Population.decs;
    PopObj   = Population.objs;
    RefDec   = PopDec(RefIndex,:);
    RefObj   = PopObj(RefIndex,:);
    Distance_Obj = pdist2(RefObj,PopObj,'chebychev'); 
    [~,OCid] = min(Distance_Obj);
    % Neighborhood in decision space
    Distance_Dec = zeros(size(RefObj,1),size(PopObj,1));
    for i = 1 : size(RefObj,1)
        for j = 1 : size(PopObj,1)
            Distance_Dec = pdist2(RefDec(i,:),PopDec(j,:),'chebychev');
        end
    end
    [~,DCid] = min(Distance_Dec);
end

function dirV = Direction_Calculation(Problem,Population,RefIndex,id,FrontNo)
    %Direction of convergence
    RefDec = Population(RefIndex).decs; 
    RN     = length(Ref);
    dirV   = zeros(size(RefDec));
    for i = 1:RN
        Neighbor = Population(id==i); 
        F_Neighbor = FrontNo(id==i);
        if ~isempty(Neighbor) && min(F_Neighbor)>1
            dirV(i,:) = mean(Neighbor(F_Neighbor==min(F_Neighbor)).decs,1)-RefDec(i,:);
        elseif rand>0.5
            dirV(i,:)  = (RefDec(i,:)-Problem.lower)./Problem.D;
        else
            dirV(i,:)  = (Problem.upper-RefDec(i,:))./Problem.D;
        end
    end
end

function Offspring = Reproduction(Problem,RefDec,cm,dirV_obj,dirV_dec) 
    RN = length(Ref);
    CN = ceil(cm*0.5*Problem.N/RN); %Size of convergence population
    DN = Problem.N-CN*RN*2; %Size of diversity population
    OffDec = zeros(CN*RN*2+DN,Problem.D);
    for i = 1:RN
        mu = RefDec(i,:); 
        %% Convergence offsprings
        for j = 1 : CN
            OffDec((i-1)*CN+j,:)       = mu + randn(1).*dirV_obj(i,:);
            OffDec(CN*RN+(i-1)*CN+j,:) = mu + randn(1).*dirV_dec(i,:);
        end
    end
    %% Diversity offsprings
    x_1   = randi(RN,1,DN);
    x_2   = randi(RN,1,DN);
    dirV  = RefDec(x_1,:)-Ref(x_2,:);
    OffDec(Problem.N-DN+1:end,:) = RefDec(x_1,:) + randn(1,DN)'.*dirV;
    %% Evaluation
    Offspring = Problem.Evaluation(OffDec);
end

function [Population,FrontNo,CrowdDis] = EnvironmentalSelection(Population,N)
% The environmental selection of NSGA-II
    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
    Next = FrontNo < MaxFNo;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;
    
    %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
end