function offspring = Local_Search(BestErrorSolution,D,CNNnet)

%% Locate the 1 in the best solution and the number of 1
IndofOne = find( BestErrorSolution.dec == 1 );
NumofOne = numel(IndofOne);

%% Difine an all-zero subpop and an all-one mask
SubPop   = zeros(NumofOne,D);
Mask     = ones(NumofOne,NumofOne);

%% Replace the 
SubPop(:,IndofOne) = eye(NumofOne);

error = zeros(NumofOne,1);

for i = 1 : NumofOne
    NewDec     = reshape(SubPop(i,:),[D,1,1]);
    error(i,1) = predict(CNNnet,NewDec);
end

IndexofSol = randperm(NumofOne);

Np = floor(NumofOne/2);

for i = 1 : Np
    
    Sel1 = IndexofSol(i);
    Sel2 = IndexofSol(i+Np);

    if error( Sel1 ) > error( Sel2 )
        Mask(i,Sel1)   = 0;
    else
        Mask(i,Sel2)   = 0;
    end
      
end

OffPop = zeros(Np,D);
OffPop(:,IndofOne) = Mask(1:Np,:);


selectIndex = randperm(Np,1);

OffDec   = OffPop(1,:);
for i = 2 : selectIndex
    OffDec = OffDec & OffPop(i,:);
end
offspring = OffDec;
