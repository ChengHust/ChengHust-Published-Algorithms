clc
clear
clear all
warning('off')

%% ScRNA-seq Data Input: ScRNAseqData Cell x Gene (The last col is label)
pro = {'Yan'};
filePath = fullfile('Dataset',  [pro{1}, '.mat']);
load(filePath);

%% Parameters Setting
k      = 4;      % Number of re-evaluated solutions in Global search
N      = 200;    % Size of population
Na     = 50;     % Number of additional solutions
pb     = 0.3;    % The small probability
MaxFEs = 1000;   % Max Number of real FEs

D = size(ScRNAseqData,2)-1;  % Number of decision variables

%% Two-phase Initialization
Arc  = Initialization(N,D,Na,ScRNAseqData);
EFEs = size(Arc,2);

%% Initial HVrecord
PopObj    = Arc.objs;
[HVrecord,~] = calHV(PopObj,[D,1]);

%% MainLoop
Surr_Flag  = floor(EFEs/100);

while EFEs < MaxFEs   

    % Environmental Selection for Population
    [Population,~,~]  = EnvironmentalSelection(Arc,N);

    % Train surrogate
    if floor(EFEs/100) >= Surr_Flag
        SurrDec     = Population.decs;
        SurrObj     = Population.objs;
        CNNnet      = CNNTraining(SurrDec,SurrObj(:,2));
        Surr_Flag   = Surr_Flag + 1;
    end

    % Global search
    PopObjs = Population.objs;
    Fitness = -PopObjs(:,2);
    Rank    = randperm(length(Population),floor(length(Population)/2)*2);
    Loser   = Rank(1:end/2);
    Winner  = Rank(end/2+1:end);
    Change  = Fitness(Loser) >= Fitness(Winner);
    Temp           = Winner(Change);
    Winner(Change) = Loser(Change);
    Loser(Change)  = Temp;
    Local_best     = BinaryCSO(Population(Loser),Population(Winner),pb,CNNnet);
    [CanSol,~,~]   = EnvironmentalSelection(Local_best,k);
    NewSolDec      = CanSol.decs;
    for kk = 1 : k
        if sum(CanSol(kk).dec) == 0
            NewSolDec(kk,randi(D)) = 1;
        end
    end
    
    % Real evaluations
    [rate,error] = Cal_Real_Function(ScRNAseqData,NewSolDec);
    Global_Data  = d_individual(NewSolDec,[rate,error]);
    EFEs         = EFEs + size(error,1);
    Arc          = [Arc,Global_Data];
   
    % Local Search
    [Arc,FrontNo,~] = EnvironmentalSelection(Arc,EFEs);
    NDPop           = Arc(FrontNo==1);
    NDPopObj        = NDPop.objs;
    [~,index]    = min(NDPopObj(:,2));
    CurrBestSol  = NDPop(index);
    Local_best   = Local_Search(CurrBestSol,D,CNNnet);

    % Real evaluations
    [rate,error] = Cal_Real_Function(ScRNAseqData,Local_best);
    Local_Data   = d_individual(Local_best, [rate,error]);
    Arc          = [Arc,Local_Data];
    EFEs         = EFEs + 1;  

    % HV result
    [Score,~] = calHV(Arc.objs,[D,1]);
    HVrecord  = [HVrecord;Score];
    
end
