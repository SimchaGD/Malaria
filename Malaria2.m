%% Detecting Malaria with CNN
% How simple can it be?
% Dataset: https://ceb.nlm.nih.gov/repositories/malaria-datasets/

clr
filepath = strcat(pwd, "\data_edited");
imds = imageDatastore(filepath, 'IncludeSubfolders',true,'LabelSource','foldernames');

trainingSize = 0.75;

rng(1)
[trainData, testData] = splitEachLabel(imds, trainingSize, 'randomize');

bag = bagOfFeatures(trainData, "VocabularySize", 20, 'BlockWidth', [32 64]);
%%
categoryClassifier = trainImageCategoryClassifier(trainData,bag);
%%
confMatrix = evaluate(categoryClassifier,testData)