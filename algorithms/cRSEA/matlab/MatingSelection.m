function MatingPool = MatingSelection(Population,Range,N)
% The mating selection of RSEAF

% Copyright 2015-2016 Cheng He

	%% Calculate the convergence of each solution
    PopObj = Population.objs;
    PopCon = Population.cons;
    PopObj = (PopObj-repmat(Range(1,:),size(PopObj,1),1))./repmat(Range(2,:)-Range(1,:),size(PopObj,1),1);
    Con = sum(PopObj.^2,2).^0.5;
% 
%     %% Calculate the radar grid of each solution
%     [Site,~] = RadarGrid(PopObj,ceil(sqrt(size(PopObj,1))));   %Grid label
%     temp    = tabulate(Site);
%     CrowdG  = temp(:,2);
%     
%     %% Binary tournament selection
     MatingPool = zeros(1,ceil(N/2)*2);
%     grids      = TournamentSelection(2,length(MatingPool),CrowdG);
%     for i = 1 : length(MatingPool)
%         current       = find(Site==grids(i));
%         if isempty(current)
%              MatingPool(i) = randi(size(PopObj,1),1);
%         else
%             parents       = current(randi(length(current),1,2));
%             [~,best]      = min(Con(parents));
%             MatingPool(i) = parents(best);
%         end
%     end
	[FrontNo,~] = NDSort(PopObj,PopCon,inf);
	Next = TournamentSelection(2,length(MatingPool),FrontNo,Con);
	MatingPool = Next;
    
end