clc; clear; close all;

fprintf('=== SVM sur MNIST ===\n');

%% Lecture images train
fid = fopen('train-images-idx3-ubyte', 'rb');
fread(fid, 1, 'int32', 'ieee-be');
nTrain = fread(fid, 1, 'int32', 'ieee-be');
rows   = fread(fid, 1, 'int32', 'ieee-be');
cols   = fread(fid, 1, 'int32', 'ieee-be');
data   = fread(fid, inf, 'uint8');
fclose(fid);
XTrain = reshape(data, rows*cols, nTrain)' / 255.0;

%% Lecture labels train
fid = fopen('train-labels-idx1-ubyte', 'rb');
fread(fid, 1, 'int32', 'ieee-be');
fread(fid, 1, 'int32', 'ieee-be');
YTrain = double(fread(fid, inf, 'uint8'));
fclose(fid);

%% Lecture images test
fid = fopen('t10k-images-idx3-ubyte', 'rb');
fread(fid, 1, 'int32', 'ieee-be');
nTest = fread(fid, 1, 'int32', 'ieee-be');
fread(fid, 1, 'int32', 'ieee-be');
fread(fid, 1, 'int32', 'ieee-be');
data  = fread(fid, inf, 'uint8');
fclose(fid);
XTest = reshape(data, rows*cols, nTest)' / 255.0;

%% Lecture labels test
fid = fopen('t10k-labels-idx1-ubyte', 'rb');
fread(fid, 1, 'int32', 'ieee-be');
fread(fid, 1, 'int32', 'ieee-be');
YTest = double(fread(fid, inf, 'uint8'));
fclose(fid);

fprintf('Train : %d | Test : %d\n', size(XTrain,1), size(XTest,1));

%% Sous-échantillon
numTrain = 5000;
numTest  = 1000;
XTr = XTrain(1:numTrain, :);
YTr = YTrain(1:numTrain);
XTe = XTest(1:numTest, :);
YTe = YTest(1:numTest);

%% SVM multi-classes (sans PCA)
fprintf('Entraînement SVM...\n');
t   = templateSVM('KernelFunction','linear','Standardize',true);
Mdl = fitcecoc(XTr, YTr, 'Learners', t);

%% Test
fprintf('Test...\n');
YPred    = predict(Mdl, XTe);
accuracy = mean(YPred == YTe) * 100;
fprintf('Taux de reconnaissance : %.2f %%\n', accuracy);

%% Matrice de confusion
figure;
cm = confusionchart(YTe, YPred);
cm.Title = 'Matrice de confusion SVM - MNIST';
cm.RowSummary = 'row-normalized';

%% Mauvaises classifications
wrong_idx = find(YPred ~= YTe);
numErrors = min(16, length(wrong_idx));
if numErrors > 0
    figure('Name','Mauvaises classifications');
    for i = 1:numErrors
        idx = wrong_idx(i);
        img = reshape(XTe(idx,:), 28, 28);
        subplot(4,4,i);
        imshow(img, []);
        title(['R:', num2str(YTe(idx)), ' P:', num2str(YPred(idx))], 'Color','r');
    end
end

%% Exemple correct
correct_idx = find(YPred == YTe);
idx = correct_idx(1);
img = reshape(XTe(idx,:), 28, 28);
figure;
imshow(img, []);
title(['Vrai : ', num2str(YTe(idx)), ' | Prédit : ', num2str(YPred(idx))]);