%% Detecting Malaria with CNN
% How simple can it be?
% Dataset: https://ceb.nlm.nih.gov/repositories/malaria-datasets/


filepath = strcat(pwd, "\data_edited");
imds = imageDatastore(filepath, 'IncludeSubfolders',true,'LabelSource','foldernames');

nImg = length(imds.Files);
I = imread(imds.Files{1});
nPixel = numel(I);
AllImageCell = zeros(nImg, nPixel);

f = waitbar(0, "Processing data");
for ii = 1:nImg
    file = imds.Files{ii};
    I = imread(file);
    AllImageCell(ii, :) = reshape(I, [1, nPixel]);
    filesplit = split(file, "\");

    waitbar(ii/nImg, f, sprintf("Processing data\n%s", filesplit{end}))
end
close(f)
disp("done")
save