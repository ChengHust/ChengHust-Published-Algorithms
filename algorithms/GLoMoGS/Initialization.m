function Archive = Initialization(N,D,Na,ScRNAseqData)

%% 1st Initialization

Dp = ceil(D/N) * N;

AllDec = [];

for i = 1 : Dp/N +1
    AllDec = [AllDec,eye(N)];
end

ind          = randi(N);
First_PopDec = AllDec(:,ind:ind+D-1);

% Real evaluations
[rate,error]   = Cal_Real_Function(ScRNAseqData,First_PopDec);  % Expensive fitness
First_Solution = d_individual(First_PopDec,[rate,error]);

%% 2nd Initialization
[~,index] = sort(error);
Second_SolDec = [];

for i = 2 : Na+1
    HalfSelIndex = index(1:ceil(N/i));
    Second_SolDec = [Second_SolDec;sum(First_PopDec(HalfSelIndex,:))];
end

% Real evaluations
[rate,error]    = Cal_Real_Function(ScRNAseqData,Second_SolDec);   % Expensive fitness
Second_Solution = d_individual(Second_SolDec,[rate,error]);


Archive  = [First_Solution,Second_Solution]; 
