function A1 = OperatorVREA(Problem,Population,V,iter,SubN,r,k)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
 
% This function is written by Jianqing Lin

    phi   = Problem.FE/(1+Problem.maxFE);
    theta = 1-(1.5*phi-0.5);

    %% Delete the solutions with same decision variables
    [~,indx,~] = unique(Population.decs,'rows');
    Population = Population(indx);

    %% Reference Solutions Selection 
    Reference  = max(Population.objs,[],1);
    NP         = size(Population,2);
    [Population,FrontNo,~] = EnvironmentalSelection(Population,min(NP,150));
    NonDomSol  = Population(FrontNo==1);

    %% Create RBF models      
    Populationdesc = Population.decs;
    Populationobjs = Population.objs;

    RBF_para = cell(1,Problem.M);
    for m = 1:Problem.M
        RBF_para{m} = RBFCreate(Populationdesc, Populationobjs(:,m), 'cubic');
    end
    
    RefPop  = Population(KrigingSelectTEST(Population.objs,V,r,phi)); 
       
    A1 = [];
    for ik = 1:size(RefPop,2)
        
        RefSol    = RefPop(ik);
        RefSolDec = RefSol.decs;
        
       %% Select k solutions to Reference Solutions
        % Cluster selection         
        [~, NeiborSolDec] = kmeans(Population.decs, k);

        % the Distance between Selection Solutions and Reference Solutions
        Neibor2Ref   = sum((NeiborSolDec-repmat(RefSolDec,k,1)).^2,2).^(0.5);
            
        % Prevent the base from being 0
        if all(Neibor2Ref) == 0
            ind = find(Neibor2Ref == 0);
            Neibor2Ref(ind) = Neibor2Ref(ind)+0.0001;
        end
        
       %% Calculate the reference directions
        direct  = (repmat(RefSolDec,k,1)-NeiborSolDec)./repmat(Neibor2Ref,1,Problem.D);
        wmax    = sum((Problem.upper-Problem.lower).^2)^(0.5)*theta;

       %% Optimize the weight variables by GA
        GAProblem.lower = zeros(1,k);
        GAProblem.upper = repmat(wmax,1,k);

        w0 = rand(SubN,k).*wmax;                                                            % Initialize the population
        [fitness,~] = fitfunc(w0,direct,NeiborSolDec,Problem,Reference,RBF_para);	        % Calculate the fitness and store the solutions

        %% Create kriging models
        Kriging_Model = dacefit(w0,fitness,'regpoly0','corrgauss',5.*ones(1,k),1e-5.*ones(1,k),10.*ones(1,k));

        %% Define the optimal solution evaluated by RBF
        GAPop = W_individual(w0,fitness);

        for it = 1 : iter
            Parent       = GAPop.decs;
            OffspringDec = GAPrime(GAProblem,Parent);
            OffspringObj = predictor(OffspringDec, Kriging_Model);
            Offspring    = W_individual(OffspringDec,OffspringObj);
            AllPop       = [GAPop,Offspring];
            [~,xxind]    = sort(AllPop.objs);
            GAPop        = AllPop(xxind(1:SubN));
        end
        
        [~,minIND]    = min(GAPop.objs);
        optsol        = GAPop(minIND);
        
        [~,OffSpring] = fitfunc(optsol.decs,direct,NeiborSolDec,Problem,Reference,RBF_para);	        % Calculate the fitness and store the solutions
        
        %% HVI Selection
        HVI_Value = [];       
        for can = 1 : k
            CanSolObj  = [NonDomSol.objs;OffSpring(can).objs];
            HVvalue    = calHV(CanSolObj,Reference) - calHV(NonDomSol.objs,Reference);
            HVI_Value  = [HVI_Value;HVvalue];
        end
        
        [~,indHV]    = max(HVI_Value);
        CandidateDec = OffSpring(indHV(1)).decs;  
        A1           = [A1;CandidateDec];

    end

    if rand > 1-phi
        A1 = GAPrime(Problem,A1);
    end

end


function [Obj,OffSpring] = fitfunc(w0,direct,NeiborSolDec,Problem,Reference,RBF_para)
    [SubN,WD] = size(w0); 
    Obj   	  = zeros(SubN,1);
    OffSpring = [];
    for i = 1 : SubN 
        PopDec    = repmat(w0(i,1:WD)',1,Problem.D).*direct(1:WD,:) + NeiborSolDec;
        PopObj    = Surrogate_Predictor(PopDec, RBF_para, Problem.M);
        OffWPop   = W_individual(PopDec,PopObj);
        
        OffSpring = [OffSpring,OffWPop];
        Obj(i)    = -calHV(OffWPop.objs,Reference);
    end
end