%% === Step 1: Load the trained model ===
load('ResNet18_AnimalClassifier_Optimized.mat');  % loads 'trainedNet'

%% === Step 2: Load saved evaluation results ===
load('ClassificationResults.mat');  % loads 'results' struct

%% === Step 3: Display summary ===
fprintf('=== Evaluation Summary ===\n');
fprintf('Test Accuracy: %.2f%%\n', results.testAccuracy * 100);
fprintf('Validation Accuracy: %.2f%%\n', results.validationAccuracy * 100);
fprintf('Training Time: %.1f minutes\n', results.trainingTime / 60);

fprintf('\nPer-class Accuracy:\n');
for i = 1:length(results.classes)
    fprintf('%s: %.2f%%\n', string(results.classes(i)), results.classAccuracy(i) * 100);
end

%% === Step 4: Confusion Matrix ===
figure('Name','Confusion Matrix','Position',[100 100 600 500]);
confusionchart(results.confusionMatrix, results.classes, ...
    'Title', 'Test Set Confusion Matrix', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
saveas(gcf, 'ConfusionMatrix.png');
%% === Step 5: Precision, Recall, F1-Score ===
cm = results.confusionMatrix;
classes = results.classes;

[precision, recall, f1score] = deal(zeros(numel(classes),1));

for i = 1:numel(classes)
    TP = cm(i,i);
    FP = sum(cm(:,i)) - TP;
    FN = sum(cm(i,:)) - TP;
    precision(i) = TP / (TP + FP);
    recall(i) = TP / (TP + FN);
    f1score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
end

fprintf('\n=== Classification Metrics ===\n');
for i = 1:numel(classes)
    fprintf('\nClass: %s\n', string(classes{i}));
    fprintf('Precision: %.2f%%\n', precision(i)*100);
    fprintf('Recall: %.2f%%\n', recall(i)*100);
    fprintf('F1 Score: %.2f%%\n', f1score(i)*100);
end

%% === Step 6: ROC Curve (One-vs-All) ===
fprintf('\n=== ROC Curve (One-vs-All) ===\n');

% Recreate test set to get YTrue and YPred scores
datasetPath = 'C:\archive\raw-img';
imds = imageDatastore(datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
requiredClasses = {'cane', 'farfalla', 'elefante'};
imds = subset(imds, ismember(imds.Labels, requiredClasses));
minCount = min(countEachLabel(imds).Count);
imds = splitEachLabel(imds, minCount, 'randomized');
imds.Labels = removecats(imds.Labels);
[~, imdsTemp] = splitEachLabel(imds, 0.75, 'randomized');
[~, imdsTest] = splitEachLabel(imdsTemp, 0.5, 'randomized');

augImdsTest = augmentedImageDatastore(trainedNet.Layers(1).InputSize(1:2), imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');

YPredScores = predict(trainedNet, augImdsTest);
YTrue = imdsTest.Labels;
classList = categories(YTrue);

% One-hot encode true labels
YTrueOneHot = zeros(numel(YTrue), numel(classList));
for i = 1:numel(YTrue)
    YTrueOneHot(i, YTrue(i) == classList) = 1;
end

% Plot ROC Curve
figure('Name', 'ROC Curve', 'Position', [100 100 800 600]);
hold on;
colors = lines(numel(classList));
legendText = cell(numel(classList),1);
for i = 1:numel(classList)
    [X, Y, ~, AUC] = perfcurve(YTrueOneHot(:,i), YPredScores(:,i), 1);
    plot(X, Y, 'LineWidth', 2, 'Color', colors(i,:));
    legendText{i} = sprintf('%s (AUC = %.2f)', classList{i}, AUC);
end
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve (One-vs-All)');
legend(legendText, 'Location', 'Best');
grid on;
hold off;
saveas(gcf, 'ROC_Curve.png');

%% === Step 7: Class-wise Bar Plot ===
figure('Name','Per-Class Metrics','Position',[100 100 800 500]);

metricsMatrix = [results.classAccuracy * 100, precision * 100, recall * 100, f1score * 100];
bar(metricsMatrix);
grid on;
title('Per-Class Evaluation Metrics');
ylabel('Percentage (%)');
legend({'Accuracy', 'Precision', 'Recall', 'F1-Score'}, 'Location', 'northoutside', 'Orientation','horizontal');
set(gca, 'XTickLabel', classes, 'XTickLabelRotation', 0, 'FontSize', 10);
saveas(gcf, 'PerClass_BarPlot.png');
