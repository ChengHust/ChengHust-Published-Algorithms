function net = CNNTraining(DataDec,DataObj)


%% Data processing
% Data transformation: row-gene col-cell
DataDec = DataDec';
DataObj = DataObj';

N = size(DataDec,2);

[trainIdx,testIdx] = crossvalind('HoldOut', N, 1/3);

TrainDec = DataDec(:,trainIdx);
TrainObj = DataObj(:,trainIdx);

TestDec = DataDec(:,testIdx);
TestObj = DataObj(:,testIdx);

%% Norm
method=@mapminmax;
[TrainDec,train_ps] = method(TrainDec,0,1);
TestDec  = method('apply',TestDec,train_ps);

[TrainObj,output_ps] = method(TrainObj,0,1);
TestObj  = method('apply',TestObj,output_ps);


%% Data re-transformation
[TrD,TrN] = size(TrainDec);
[TeD,TeN] = size(TestDec);

trainD  = reshape(TrainDec,[TrD,1,1,TrN]);
testD   = reshape(TestDec, [TeD,1,1,TeN]);
targetD       = TrainObj;
targetD_test  = TestObj;


%% Model Building
layers = [
    imageInputLayer([TeD 1 1])

    % First convolutional layer
    convolution2dLayer([9, 1], 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([2, 1], 'Stride', [2, 1])
    
    % The second convolutional layer
    convolution2dLayer([9, 1], 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([2, 1], 'Stride', [2, 1])

    
    % Full conn
    fullyConnectedLayer(128)
    reluLayer
    
    % regressionLayer
    fullyConnectedLayer(1)
    regressionLayer
];


%% Model training and testing
options = trainingOptions('adam', ... 
                          'MaxEpochs',20, ...
                          'MiniBatchSize',64, ...
                          'InitialLearnRate',0.001, ...
                          'GradientThreshold',1, ...
                          'Verbose',false, ...
                          'Plots','none',...
                          'ValidationData',{testD,targetD_test'});

% training
net = trainNetwork(trainD,targetD',layers,options);

% predict
YPred = predict(net,testD);

% n x 1 to 1 x n double
YPred = double(YPred');

% Denorm
predict_value  =  method('reverse',YPred,output_ps);predict_value=double(predict_value);
true_value     =  method('reverse',targetD_test,output_ps);true_value=double(true_value);
