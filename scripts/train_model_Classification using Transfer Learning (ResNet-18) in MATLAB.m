%% Step 1: Enhanced Dataset Setup
datasetPath = C:/your_dataset_folder/raw-img';
tic; % Start timing

% Load dataset with labels from folders
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Filter only required 3 classes
requiredClasses = {'cane', 'farfalla', 'elefante'};
imds = subset(imds, ismember(imds.Labels, requiredClasses));
disp("Original class counts:"); 
countEachLabel(imds)

% Balance dataset to minimum class size
minCount = min(countEachLabel(imds).Count);
imds = splitEachLabel(imds, minCount, 'randomized');
imds.Labels = removecats(imds.Labels);  % Clean unused labels

disp("Balanced class counts:"); 
countEachLabel(imds)
fprintf('Using %d samples per class\n', minCount);

%% Step 2: Optimized Data Split
% Use stratified split for balanced training/validation/test
[imdsTrain, imdsTemp] = splitEachLabel(imds, 0.75, 'randomized');  % 75% train
[imdsValidation, imdsTest] = splitEachLabel(imdsTemp, 0.5, 'randomized');  % 12.5% val, 12.5% test

fprintf('\nDataset Split:\n');
fprintf('Train: %d samples\n', numel(imdsTrain.Files));
fprintf('Validation: %d samples\n', numel(imdsValidation.Files));
fprintf('Test: %d samples\n', numel(imdsTest.Files));

%% Step 3: Load and Modify ResNet-18 for Transfer Learning
net = resnet18;
inputSize = net.Layers(1).InputSize;

% Create layer graph and replace classification layers
lgraph = layerGraph(net);

% Remove final layers
lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});

% Add new classification layers with dropout for better generalization
newLayers = [
    dropoutLayer(0.5, 'Name', 'dropout')
    fullyConnectedLayer(3, 'Name', 'fc3', ...
        'WeightLearnRateFactor', 10, ...  % Higher learning rate for new layers
        'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'pool5', 'dropout');

%% Step 4: Enhanced Data Augmentation
% More aggressive augmentation for better generalization
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandYReflection', false, ...  % Animals should maintain orientation
    'RandRotation', [-15, 15], ...  % Moderate rotation
    'RandXTranslation', [-20, 20], ...
    'RandYTranslation', [-20, 20], ...
    'RandScale', [0.9, 1.1], ...  % Slight scaling
    'RandXShear', [-5, 5], ...    % Small shear for variety
    'RandYShear', [-5, 5]);

% Create augmented datastores
augImdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', imageAugmenter, ...
    'ColorPreprocessing', 'gray2rgb');

augImdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation, ...
    'ColorPreprocessing', 'gray2rgb');

augImdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');

%% Step 5: Optimized Training Options
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...      % Slightly higher for faster convergence
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...     % Drop LR by half
    'LearnRateDropPeriod', 8, ...       % Every 8 epochs
    'MaxEpochs', 20, ...
    'MiniBatchSize', 64, ...            % Larger batch size for stability
    'Shuffle', 'every-epoch', ...
    'ValidationData', augImdsValidation, ...
    'ValidationFrequency', 20, ...      % Less frequent validation for speed
    'ValidationPatience', 5, ...        % Early stopping
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');    % Let MATLAB choose best option

%% Step 6: Train the Network
fprintf('\n=== Starting Training ===\n');
trainingStart = tic;
trainedNet = trainNetwork(augImdsTrain, lgraph, options);
trainingTime = toc(trainingStart);
fprintf('Training completed in %.1f minutes\n', trainingTime/60);

%% Step 7: Comprehensive Evaluation
fprintf('\n=== Model Evaluation ===\n');

% Test accuracy
YPred = classify(trainedNet, augImdsTest);
YTrue = imdsTest.Labels;
accuracy = mean(YPred == YTrue);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% Validation accuracy
YPredVal = classify(trainedNet, augImdsValidation);
YTrueVal = imdsValidation.Labels;
valAccuracy = mean(YPredVal == YTrueVal);
fprintf('Validation Accuracy: %.2f%%\n', valAccuracy * 100);

% Per-class accuracy
classes = categories(YTrue);
classAccuracy = zeros(length(classes), 1);
for i = 1:length(classes)
    classIdx = YTrue == classes{i};
    classAccuracy(i) = mean(YPred(classIdx) == YTrue(classIdx));
    fprintf('%s Accuracy: %.2f%%\n', classes{i}, classAccuracy(i) * 100);
end

%% Step 8: Visualization
% Confusion Matrix
figure('Position', [100, 100, 800, 600]);
subplot(2, 2, 1);
confusionchart(YTrue, YPred, 'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
title('Test Set Confusion Matrix');

% Sample predictions
subplot(2, 2, [2, 4]);
numSamples = 9;
sampleIdx = randperm(numel(imdsTest.Files), numSamples);
for i = 1:numSamples
    subplot(3, 3, i);
    I = readimage(imdsTest, sampleIdx(i));
    imshow(I);
    
    % Get prediction for this image
    predLabel = YPred(sampleIdx(i));
    trueLabel = YTrue(sampleIdx(i));
    
    % Color code: green for correct, red for incorrect
    if predLabel == trueLabel
        titleColor = 'green';
        accuracy_symbol = '✓';
    else
        titleColor = 'red';
        accuracy_symbol = '✗';
    end
    
    title(sprintf('%s Pred: %s %s', accuracy_symbol, predLabel, ''), ...
        'Color', titleColor, 'FontSize', 8);
end
sgtitle('Sample Predictions (Green=Correct, Red=Incorrect)');

%% Step 9: Save Model and Results
% Save the trained model
save('ResNet18_AnimalClassifier_Optimized.mat', 'trainedNet');

% Save evaluation results
results = struct();
results.testAccuracy = accuracy;
results.validationAccuracy = valAccuracy;
results.classAccuracy = classAccuracy;
results.classes = classes;
results.trainingTime = trainingTime;
results.confusionMatrix = confusionmat(YTrue, YPred);

save('ClassificationResults.mat', 'results');

%% Step 10: Performance Summary
totalTime = toc;
fprintf('\n=== Performance Summary ===\n');
fprintf('Total Runtime: %.1f minutes\n', totalTime/60);
fprintf('Training Time: %.1f minutes\n', trainingTime/60);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Target Met: %s\n', string(accuracy >= 0.90));
fprintf('Model saved as: ResNet18_AnimalClassifier_Optimized.mat\n');

% Check if all requirements are met
if accuracy >= 0.90
    fprintf('SUCCESS: 90%% accuracy target achieved!\n');
else
    fprintf('Target not met. Consider:\n');
    fprintf('- Increasing epochs\n');
    fprintf('- Adjusting learning rate\n');
    fprintf('- More data augmentation\n');
end

if trainingTime <= 25*60  % 25 minutes in seconds
    fprintf('Training time requirement met!\n');
else
    fprintf('Training took longer than 25 minutes\n');
end
