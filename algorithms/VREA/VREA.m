classdef VREA < ALGORITHM
% <multi/many> <real/integer> <expensive>
% 
% r      ---  5  --- The number of reference solutions
% k      ---  15  --- The number of directly solutions
% iter   ---  50 --- Maximum number of iterations
% SubN   ---  50 --- The number of population

%------------------------------- Reference --------------------------------
% J. Lin, C. He, Y. Tian and L. Pan. Variable Reconstruction for Evolutionary 
% Expensive Large-Scale Multiobjective Optimization and Its Application on 
% Aerodynamic Design. IEEE/CAA Journal of Automatica Sinica, vol. 12, no. 4,
% pp. 719-733 2025

%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Jianqing Lin

    methods

        function main(Algorithm, Problem)
            %% Parameter setting
            [r,k,iter,SubN] = Algorithm.ParameterSet(5,5,50,50);
            [V0,~] = UniformPoint(SubN,Problem.M);
	        V      = V0;
            
            %% Generate initial population
            PopDec = repmat((Problem.upper - Problem.lower),Problem.N, 1) .* lhsdesign(Problem.N, Problem.D) + repmat(Problem.lower, Problem.N, 1);  % lhs design
            Arc    = Problem.Evaluation(PopDec);
            
            %% Optimization
            while Algorithm.NotTerminated(Arc)
                A1  = OperatorVREA(Problem,Arc,V,iter,SubN,r,k);
                A1  = Problem.Evaluation(A1);
                V(1:size(V0,1),:) = V0.*repmat(max(A1.objs,[],1)-min(A1.objs,[],1),size(V0,1),1);
                Arc = [Arc,A1];
            end
        end
    end
end