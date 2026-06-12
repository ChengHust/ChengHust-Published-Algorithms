function Population = PFExtend(Global,Population,Operator,mode)
% The second step of CLEA, extend the searching along the PS (PF)

%--------------------------------------------------------------------------
% The copyright of the PlatEMO belongs to the BIMK Group. You are free to
% use the PlatEMO for research purposes. All publications which use this
% platform or any code in the platform should acknowledge the use of
% "PlatEMO" and reference "Ye Tian, Ran Cheng, Xingyi Zhang, and Yaochu
% Jin, PlatEMO: A MATLAB Platform for Evolutionary Multi-Objective
% Optimization, 2016".
%--------------------------------------------------------------------------

% Copyright (c) 2016-2017 Cheng He
	Gmax = ceil((Global.evaluation-Global.evaluated)/Global.N);
    switch mode
        case 1 
            Population = subNSGAII(Global,Population,Operator,Gmax);
        case 2
            Population = subMOEADDE(Global,Population,Operator,Gmax);
        case 3
            Population = subCMOPSO(Global,Population,Operator,Gmax);
        case 4
            Population = subSMSEMOA(Global,Population,Operator,Gmax);
        case 5
            Population = subSPEA2(Global,Population,Operator,Gmax);
        case 6
            Population = subSMPSO(Global,Population,Operator,Gmax);
        otherwise
            Population = subIBEA(Global,Population,Operator,Gmax); 
    end
end