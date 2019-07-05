%% Detecting Malaria with CNN
% How simple can it be?
% Dataset: https://ceb.nlm.nih.gov/repositories/malaria-datasets/

clr
filepath = strcat(pwd, "\data");
imds = imageDatastore(filepath, 'IncludeSubfolders',true,'LabelSource','foldernames');
%%
labelCount = countEachLabel(imds);
subplot(1, 2, 1)
imshow(imds.Files{1})
title("No Malaria")


subplot(1, 2, 2)
imshow(imds.Files{13780})
title("Malaria")
disp(labelCount)
%%
 