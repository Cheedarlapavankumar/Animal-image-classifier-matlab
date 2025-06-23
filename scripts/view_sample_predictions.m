clc; clear; close all;

%% === Load Trained Model ===
load('ResNet18_AnimalClassifier_Optimized.mat');

%% === Reconstruct Test Set ===
datasetPath = 'C:\archive\raw-img';
imds = imageDatastore(datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imds = subset(imds, ismember(imds.Labels, {'cane','farfalla','elefante'}));
minCount = min(countEachLabel(imds).Count);
imds = splitEachLabel(imds, minCount, 'randomized');
imds.Labels = removecats(imds.Labels);

% Create test split
[~, imdsTemp] = splitEachLabel(imds, 0.75, 'randomized');
[~, imdsTest] = splitEachLabel(imdsTemp, 0.5, 'randomized');

% Create augmented datastore for test set
augImdsTest = augmentedImageDatastore(trainedNet.Layers(1).InputSize(1:2), imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');

%% === Predict and Show Random Sample Images ===
YPred = classify(trainedNet, augImdsTest);
YTrue = imdsTest.Labels;

numSamples = 9;
sampleIdx = randperm(numel(imdsTest.Files), numSamples);

fig = figure('Name','Random Predictions','Position',[100 100 1000 700]);
sgtitle('Sample Predictions (Green = Correct, Red = Incorrect)', 'FontSize', 14);

for i = 1:numSamples
    subplot(3, 3, i);
    I = readimage(imdsTest, sampleIdx(i));
    imshow(I);

    predLabel = YPred(sampleIdx(i));
    trueLabel = YTrue(sampleIdx(i));

    if predLabel == trueLabel
        titleColor = 'green'; mark = '✓';
    else
        titleColor = 'red'; mark = '✗';
    end

    title(sprintf('%s Pred: %s', mark, string(predLabel)), ...
        'Color', titleColor, 'FontSize', 10);
end

%% === Save Figure Automatically ===
timestamp = datestr(now, 'dd-mmm-yyyy_HH-MM-SS');
saveas(fig, ['SamplePredictions_' timestamp '.png']);
fprintf('Saved image as: SamplePredictions_%s.png\n', timestamp);
