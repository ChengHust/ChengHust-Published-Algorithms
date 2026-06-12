function [V1,V2] = ReferenceVectorAdaptation(PopObj,V1,V2)
% Reference vector adaption strategy of two reference vector sets

%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    PopObj = max(PopObj,[],1)-min(PopObj,[],1);
    V1 	   = V1.*repmat(PopObj,size(V1,1),1); 
    V2     = V2.*repmat(PopObj,size(V2,1),1); 
end