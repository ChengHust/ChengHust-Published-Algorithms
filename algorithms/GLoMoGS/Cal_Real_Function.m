function [rate,error] = Cal_Real_Function(Originaldata,indX)

N = size(indX,1);
D = size(Originaldata,2)-1;

rate  = zeros(N,1);
error = zeros(N,1);


for i = 1 : N 
    
    ind  = indX(i,:);
    
    %% Data processing
    data  = Originaldata(:,1:end-1);
    label = Originaldata(:,end);
    scale = size(data,1);


    SelectInd = find(ind==1);
    
    data  = data(:,SelectInd);
    
    % 10-fold
    nFold    = 10;
    indices = crossvalind('Kfold', scale, nFold);

    Accuracy = 0;
    for nF = 1 : nFold 
    
        % train and test indices
        testIndices  = (indices == nF);
        trainIndices   = ~testIndices;
       
        % train and train label
        trainData  = data(trainIndices, :);
        trainLabel = label(trainIndices, :);
    
        % test and test label
        testData = data(testIndices, :);
        testLabel = label(testIndices, :);
    
        %% KNN
        kNNClassifier = fitcknn(trainData, trainLabel, 'NumNeighbors', 5);
        Prelabel = predict(kNNClassifier,testData);
        Acc = sum(Prelabel==testLabel)/length(testLabel);
        Accuracy = Accuracy + Acc;
    end
    Acc = Accuracy / nFold;
    rate(i,:)  = sum(ind);
    error(i,:) = 1 - Acc;

end
