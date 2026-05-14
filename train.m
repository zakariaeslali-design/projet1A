clear; clc; close all;

%% ================= LOAD DATA =================
X_train = loadMNISTImages('train-images.idx3-ubyte')';
y_train = loadMNISTLabels('train-labels.idx1-ubyte');

X_test = loadMNISTImages('t10k-images.idx3-ubyte')';
y_test = loadMNISTLabels('t10k-labels.idx1-ubyte');

% MEME SUBSET QUE KNN
X_train = X_train(1:10000,:);
y_train = y_train(1:10000);

X_test = X_test(1:2000,:);
y_test = y_test(1:2000);

% RESHAPE POUR CNN
XTrain = reshape(X_train',28,28,1,[]);
XTest  = reshape(X_test',28,28,1,[]);

% NORMALISATION
XTrain = im2single(XTrain);
XTest  = im2single(XTest);

% LABELS
YTrain = categorical(y_train);
YTest  = categorical(y_test);

%% ================= CNN =================
layers = [
    imageInputLayer([28 28 1],'Normalization','none')

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    fullyConnectedLayer(128)
    reluLayer

    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs',5, ... % plus rapide pour comparaison
    'MiniBatchSize',128, ...
    'Verbose',false);

%% ================= TRAIN =================
net = trainNetwork(XTrain, YTrain, layers, options);

%% ================= TEST =================
YPred_cat = classify(net, XTest);

% CONVERTIR EN DOUBLE (comme KNN)
YPred = double(YPred_cat) - 1;
YTrue = double(YTest) - 1;

accuracy = mean(YPred == YTrue) * 100;
fprintf('Taux de reconnaissance CNN : %.2f %%\n', accuracy);

%% ================= MATRICE DE CONFUSION =================
figure;
confusionchart(YTrue, YPred);
title('Matrice de confusion CNN - MNIST');

%% ================= IMAGE TEST =================
i = 3;
img = reshape(X_test(i,:), 28, 28);

figure;
imshow(img', []);
title(['Vrai : ', num2str(y_test(i)), ...
       ' | Prédit : ', num2str(YPred(i))]);

%% ================= ERREURS =================
wrong_idx = find(YPred ~= YTrue);
numErrors = min(16, length(wrong_idx));

if numErrors > 0
    figure('Name','Erreurs CNN');
    for j = 1:numErrors
        idx = wrong_idx(j);
        img_err = reshape(X_test(idx,:), 28, 28);
        subplot(4,4,j);
        imshow(img_err', []);
        title(['R: ', num2str(y_test(idx)), ...
               ' | P: ', num2str(YPred(idx))], 'Color','r');
    end
end