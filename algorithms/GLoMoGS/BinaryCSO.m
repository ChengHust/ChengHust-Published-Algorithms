function Offspring = BinaryCSO(Loser,Winner,pb,CNNnet)

    %% Parameter setting
    LoserDec  = Loser.decs;
    WinnerDec = Winner.decs;
    [N,D]     = size(LoserDec);

    p1  = sum(WinnerDec)./max(sum(WinnerDec));

    %% Competitive swarm optimizer
    r1 = repmat(rand(N,1),1,D);
    r2 = rand(N,D); 

    R1 = r1 <= repmat(p1,N,1);
    R2 = r2 <= repmat(pb,N,D);

    OffDec = LoserDec - (WinnerDec&LoserDec) + R1.*(WinnerDec&LoserDec) + R2.*(WinnerDec-LoserDec);
    
    [N,D]  = size(OffDec);
    Lower  = repmat(zeros(1,D),N,1);
    Upper  = repmat(ones(1,D),N,1);
    OffDec = max(min(OffDec,Upper),Lower);

    OffspringDec = zeros(N,D); 
    OffspringObj = zeros(N,2);
   

    %% mutation

    for i = 1 : N
        
        if numel(find(OffDec(i,:) == 1))<=1
            MutaDec = zeros(1,D);
            MutaDec(1,randi(D)) = 1;
        elseif numel(find(OffDec(i,:) == 1))>1 && numel(find(OffDec(i,:) == 1))<10
            MutaDec  = OffDec(i,:);
            randone  = find( MutaDec == 1);
            randDel  = randi(numel(randone));
            MutaDec(:,randone(randDel)) = 0;
        else
            MutaDec  = OffDec(i,:);
            randone  = find( MutaDec == 1);
            randDel  = randperm(numel(randone),floor(numel(randone)/10));
            MutaDec(:,randone(randDel)) = 0;
        end
        
        OffDec_reshape = reshape(OffDec(i,:),[D,1,1]);
        MutDec_reshape = reshape(MutaDec,[D,1,1]);
       
        error0  = predict(CNNnet,OffDec_reshape);
        error1  = predict(CNNnet,MutDec_reshape);
     
        if error1 <= error0
            OffspringDec(i,:) = MutaDec;
            OffspringObj(i,1) = sum(MutaDec);
            OffspringObj(i,2) = error1;
        else
            OffspringDec(i,:) = OffDec(i,:);
            OffspringObj(i,1) = sum(OffDec(i,:));
            OffspringObj(i,2) = error0;
        end

    end

    Offspring = d_individual(OffspringDec,OffspringObj);
end
