%% ========================================================================
% PROJECT 29: Retinal Image Enhancement Using CLAHE, Retinex, and Deep Learning
% Architecture Reference: Explainable MINet-ViT (Alanazi et al., DOI: 10.1002/ima.70245)
% Stage 1 (MATLAB Online Optimized): Real APTOS 2019 Batch Preprocessing,
% Classical Enhancement (CLAHE, MSRCR Retinex) & Anscombe VST + Auto-ZIP
% =========================================================================

clear; clc; close all;

%% 1. CONFIGURATION & DIRECTORY SETUP (OPTIMIZED FOR MATLAB ONLINE)
% Automatically detect your /MATLAB Drive/raw_fundus_images/train_images folder
candidateDirs = {
    '/MATLAB Drive/raw_fundus_images/train_images/train_images', ...
    '/MATLAB Drive/raw_fundus_images/train_images', ...
    './raw_fundus_images/train_images/train_images', ...
    './raw_fundus_images/train_images', ...
    '/MATLAB Drive/raw_fundus_images', ...
    './raw_fundus_images'
};

cfg.inputDir = './raw_fundus_images';
for k = 1:numel(candidateDirs)
    if exist(candidateDirs{k}, 'dir')
        testPng = dir(fullfile(candidateDirs{k}, '*.png'));
        testJpg = dir(fullfile(candidateDirs{k}, '*.jpg'));
        if ~isempty(testPng) || ~isempty(testJpg)
            cfg.inputDir = candidateDirs{k};
            break;
        end
    end
end

cfg.outputBaseDir      = './processed_datasets';  % Export folder
cfg.zipFileName        = 'processed_datasets.zip';% Auto-generated ZIP for 1-click download
cfg.targetSize         = [256, 256];              % Standard 256x256 for fast MINet & CNN-ViT
cfg.maxImagesToProcess = 150;                     % Cap at 150 real scans (~90 sec on MATLAB Online)
cfg.saveMatTensors     = false;                   % False saves 90% disk quota; PNGs are lossless enough

% Subdirectories for exported datasets & figures
cfg.dirCropped    = fullfile(cfg.outputBaseDir, 'dataset_cropped');
cfg.dirCLAHE      = fullfile(cfg.outputBaseDir, 'dataset_clahe');
cfg.dirRetinex    = fullfile(cfg.outputBaseDir, 'dataset_retinex');
cfg.dirVST        = fullfile(cfg.outputBaseDir, 'dataset_vst');
cfg.dirVST_Den    = fullfile(cfg.outputBaseDir, 'dataset_vst_restored');
cfg.dirFigures    = fullfile(cfg.outputBaseDir, 'evaluation_figures');

% Algorithm Hyperparameters
cfg.fovThreshold   = 12;                          % Gray-level threshold for black border masking
cfg.claheClip      = 0.015;                       % CLAHE ClipLimit
cfg.claheTiles     = [8, 8];                      % CLAHE tile grid
cfg.retinexScales  = [15, 80, 200];               % MSRCR Gaussian scales
cfg.retinexWeights = [1/3, 1/3, 1/3];             % Equal scale weighting
cfg.msrcrAlpha     = 125.0;                       % Color restoration nonlinearity factor
cfg.msrcrBeta      = 46.0;                        % Color restoration gain constant
cfg.msrcrGain      = 1.0;                         % Final dynamic range gain
cfg.msrcrOffset    = 0.0;                         % Final dynamic range offset

% Create output directories
outDirs = {cfg.dirCropped, cfg.dirCLAHE, cfg.dirRetinex, cfg.dirVST, cfg.dirVST_Den, cfg.dirFigures};
for i = 1:numel(outDirs)
    if ~exist(outDirs{i}, 'dir')
        mkdir(outDirs{i});
    end
end

% Gather image files from the detected folder
imgExtensions = {'*.png', '*.jpg', '*.jpeg', '*.tif', '*.bmp'};
fileList = [];
for i = 1:numel(imgExtensions)
    fileList = [fileList; dir(fullfile(cfg.inputDir, imgExtensions{i}))]; %#ok<AGROW>
end

if isempty(fileList)
    fprintf('[WARN] No images found in "%s". Generating 3 synthetic phantoms as fallback...\n', cfg.inputDir);
    if ~exist(cfg.inputDir, 'dir'), mkdir(cfg.inputDir); end
    generateSyntheticFundusDataset(cfg.inputDir, 3);
    fileList = dir(fullfile(cfg.inputDir, '*.png'));
end

totalAvailable = numel(fileList);
% Uniformly sample across the sorted list so we get a diverse spread of APTOS IDs
if totalAvailable > cfg.maxImagesToProcess
    rng(42);
    sampleIndices = round(linspace(1, totalAvailable, cfg.maxImagesToProcess));
    fileList = fileList(sampleIndices);
end

numImages = numel(fileList);
fprintf('[INFO] Input Folder: "%s" (Found %d total images).\n', cfg.inputDir, totalAvailable);
fprintf('[INFO] Processing %d real fundus images at [%dx%d] resolution...\n\n', ...
    numImages, cfg.targetSize(1), cfg.targetSize(2));

%% 2. BATCH PROCESSING & QUANTITATIVE LOGGING
imageNames = cell(numImages, 1);
methods    = {'Raw_Cropped', 'CLAHE_LAB', 'MSRCR_Retinex', 'Anscombe_VST_Domain', 'VST_Baseline_Restored'};
numMethods = numel(methods);

metricsLog = struct();
for m = 1:numMethods
    mName = methods{m};
    metricsLog.(mName).PSNR      = zeros(numImages, 1);
    metricsLog.(mName).SSIM      = zeros(numImages, 1);
    metricsLog.(mName).RMS_Contr = zeros(numImages, 1);
    metricsLog.(mName).Michelson = zeros(numImages, 1);
    metricsLog.(mName).Entropy   = zeros(numImages, 1);
end

demoData = struct();

for idx = 1:numImages
    fileName = fileList(idx).name;
    [~, baseName, ~] = fileparts(fileName);
    imageNames{idx} = fileName;
    
    if idx == 1 || mod(idx, 15) == 0 || idx == numImages
        fprintf('   Processing [%3d/%3d]: %s ...\n', idx, numImages, fileName);
    end
    
    % Read raw fundus image
    rawImg = imread(fullfile(cfg.inputDir, fileName));
    if size(rawImg, 3) == 1
        rawImg = repmat(rawImg, [1, 1, 3]);
    end
    
    % ---------------------------------------------------------------------
    % STEP 1: Circular FOV Masking, Bounding-Box Cropping & Bicubic Resize
    % ---------------------------------------------------------------------
    [croppedRGB, fovMask] = preprocessRetinalFOV(rawImg, cfg.targetSize, cfg.fovThreshold);
    
    % ---------------------------------------------------------------------
    % STEP 2: Variance-Stabilizing Transform (Anscombe Forward & Inverse)
    % ---------------------------------------------------------------------
    imgDouble255 = double(croppedRGB);
    vstForward   = forwardAnscombeVST(imgDouble255);
    
    vstTheoryMin = 2 * sqrt(0 + 3/8);
    vstTheoryMax = 2 * sqrt(255 + 3/8);
    vstNorm      = (vstForward - vstTheoryMin) / (vstTheoryMax - vstTheoryMin);
    vstNorm      = min(max(vstNorm, 0), 1);
    vstUint8     = im2uint8(vstNorm .* repmat(double(fovMask), [1, 1, 3]));
    
    vstDenoised = zeros(size(vstForward));
    for ch = 1:3
        vstDenoised(:,:,ch) = imgaussfilt(vstForward(:,:,ch), 0.8);
    end
    vstRestoredDouble = inverseAnscombeVST(vstDenoised, 'exact_unbiased');
    vstRestoredUint8  = uint8(min(max(vstRestoredDouble, 0), 255)) .* uint8(repmat(fovMask, [1, 1, 3]));
    
    % ---------------------------------------------------------------------
    % STEP 3A: Classical Baseline 1 - LAB Space CLAHE
    % ---------------------------------------------------------------------
    claheUint8 = enhanceCLAHE_LAB(croppedRGB, fovMask, cfg.claheClip, cfg.claheTiles);
    
    % ---------------------------------------------------------------------
    % STEP 3B: Classical Baseline 2 - Multi-Scale Retinex (MSRCR)
    % ---------------------------------------------------------------------
    retinexUint8 = enhanceMSRCR(croppedRGB, fovMask, cfg.retinexScales, ...
                                cfg.retinexWeights, cfg.msrcrAlpha, cfg.msrcrBeta, ...
                                cfg.msrcrGain, cfg.msrcrOffset);
                            
    % ---------------------------------------------------------------------
    % STEP 4: Export Preprocessed Images to Disk (Keeping exact APTOS ID stem)
    % ---------------------------------------------------------------------
    imwrite(croppedRGB,       fullfile(cfg.dirCropped, [baseName, '_cropped.png']));
    imwrite(claheUint8,       fullfile(cfg.dirCLAHE,   [baseName, '_clahe.png']));
    imwrite(retinexUint8,     fullfile(cfg.dirRetinex, [baseName, '_retinex.png']));
    imwrite(vstUint8,         fullfile(cfg.dirVST,     [baseName, '_vst.png']));
    imwrite(vstRestoredUint8, fullfile(cfg.dirVST_Den, [baseName, '_vst_restored.png']));
    
    if cfg.saveMatTensors || idx <= 5
        % Save .mat tensor for the first 5 samples for verification without bloating ZIP size
        save(fullfile(cfg.dirVST, [baseName, '_vst_tensor.mat']), 'vstForward', 'fovMask');
    end
    
    % ---------------------------------------------------------------------
    % STEP 5: Compute Quantitative DIP Metrics
    % ---------------------------------------------------------------------
    candidates = {croppedRGB, claheUint8, retinexUint8, vstUint8, vstRestoredUint8};
    for m = 1:numMethods
        mName = methods{m};
        q = computeDIPMetrics(croppedRGB, candidates{m}, fovMask);
        metricsLog.(mName).PSNR(idx)      = q.PSNR;
        metricsLog.(mName).SSIM(idx)      = q.SSIM;
        metricsLog.(mName).RMS_Contr(idx) = q.RMS_Contrast;
        metricsLog.(mName).Michelson(idx) = q.Michelson_Contrast;
        metricsLog.(mName).Entropy(idx)   = q.Entropy;
    end
    
    if idx == 1
        demoData.name        = fileName;
        demoData.raw         = rawImg;
        demoData.cropped     = croppedRGB;
        demoData.mask        = fovMask;
        demoData.clahe       = claheUint8;
        demoData.retinex     = retinexUint8;
        demoData.vst         = vstUint8;
        demoData.vstRestored = vstRestoredUint8;
    end
end

%% 3. DISPLAY & EXPORT QUANTITATIVE EVALUATION SUMMARY TABLE
fprintf('\n=========================================================================================\n');
fprintf('     STAGE 1 REAL APTOS 2019 DIP EVALUATION SUMMARY (Averaged over N = %d scans)        \n', numImages);
fprintf('=========================================================================================\n');
fprintf('%-24s | %-10s | %-8s | %-14s | %-14s | %-10s\n', ...
    'Method', 'PSNR (dB)', 'SSIM', 'RMS Contrast', 'Michelson', 'Entropy');
fprintf('-----------------------------------------------------------------------------------------\n');

summaryMatrix = zeros(numMethods, 5);
for m = 1:numMethods
    mName = methods{m};
    meanPSNR = mean(metricsLog.(mName).PSNR);
    meanSSIM = mean(metricsLog.(mName).SSIM);
    meanRMS  = mean(metricsLog.(mName).RMS_Contr);
    meanMich = mean(metricsLog.(mName).Michelson);
    meanEnt  = mean(metricsLog.(mName).Entropy);
    
    summaryMatrix(m, :) = [meanPSNR, meanSSIM, meanRMS, meanMich, meanEnt];
    if strcmp(mName, 'Raw_Cropped')
        psnrStr = 'Inf (Ref)';
    else
        psnrStr = sprintf('%6.2f', meanPSNR);
    end
    fprintf('%-24s | %-10s | %8.4f | %14.4f | %14.4f | %10.4f\n', ...
        mName, psnrStr, meanSSIM, meanRMS, meanMich, meanEnt);
end
fprintf('=========================================================================================\n\n');

summaryTable = array2table(summaryMatrix, ...
    'VariableNames', {'PSNR_dB', 'SSIM', 'RMS_Contrast', 'Michelson_Contrast', 'Information_Entropy'}, ...
    'RowNames', methods);
writetable(summaryTable, fullfile(cfg.outputBaseDir, 'Stage1_Quantitative_Metrics.csv'), 'WriteRowNames', true);

%% 4. VISUAL DEMONSTRATION & AUTOMATIC ZIP CREATION FOR DOWNLOAD
plotStage1Comparison(demoData, metricsLog, cfg.dirFigures);

fprintf('[INFO] Creating "%s" for 1-click download from MATLAB Online...\n', cfg.zipFileName);
zip(cfg.zipFileName, cfg.outputBaseDir);
fprintf('[SUCCESS] Stage 1 complete! Right-click "%s" in MATLAB Online and click Download.\n', cfg.zipFileName);


%% ========================================================================
%                        LOCAL MODULAR FUNCTIONS
% =========================================================================

function [croppedImg, fovMaskResize] = preprocessRetinalFOV(rawImg, targetSize, thresh)
    if size(rawImg, 3) == 3
        grayImg = rgb2gray(rawImg);
    else
        grayImg = rawImg;
    end
    bwMask = grayImg > thresh;
    bwMask = imfill(bwMask, 'holes');
    bwMask = bwareaopen(bwMask, 500);
    bwMask = imclose(bwMask, strel('disk', 5));
    
    cc = bwconncomp(bwMask);
    if cc.NumObjects > 0
        numPixels = cellfun(@numel, cc.PixelIdxList);
        [~, largestIdx] = max(numPixels);
        cleanMask = false(size(bwMask));
        cleanMask(cc.PixelIdxList{largestIdx}) = true;
    else
        cleanMask = true(size(bwMask));
    end
    
    stats = regionprops(cleanMask, 'BoundingBox');
    if ~isempty(stats)
        bbox = round(stats(1).BoundingBox);
        x1 = max(1, bbox(1));
        y1 = max(1, bbox(2));
        x2 = min(size(rawImg, 2), x1 + bbox(3) - 1);
        y2 = min(size(rawImg, 1), y1 + bbox(4) - 1);
        subImg  = rawImg(y1:y2, x1:x2, :);
        subMask = cleanMask(y1:y2, x1:x2);
    else
        subImg  = rawImg;
        subMask = cleanMask;
    end
    
    [h, w, c] = size(subImg);
    maxDim = max(h, w);
    padTop  = floor((maxDim - h) / 2); padBot  = ceil((maxDim - h) / 2);
    padLeft = floor((maxDim - w) / 2); padRight = ceil((maxDim - w) / 2);
    
    paddedImg = zeros(maxDim, maxDim, c, 'like', rawImg);
    for k = 1:c
        paddedImg(:, :, k) = padarray(padarray(subImg(:, :, k), [padTop, padLeft], 0, 'pre'), ...
                                      [padBot, padRight], 0, 'post');
    end
    paddedMask = padarray(padarray(subMask, [padTop, padLeft], false, 'pre'), ...
                          [padBot, padRight], false, 'post');
    
    croppedImg    = imresize(paddedImg, targetSize, 'bicubic');
    fovMaskResize = imresize(paddedMask, targetSize, 'nearest');
    fovMaskResize = imerode(fovMaskResize, strel('disk', 2));
    croppedImg    = croppedImg .* uint8(repmat(fovMaskResize, [1, 1, 3]));
end

function vstOut = forwardAnscombeVST(z)
    z = max(double(z), 0);
    vstOut = 2.0 .* sqrt(z + (3.0 / 8.0));
end

function zHat = inverseAnscombeVST(D, method)
    if nargin < 2, method = 'exact_unbiased'; end
    D = double(D);
    switch lower(method)
        case 'algebraic'
            zHat = (D ./ 2.0).^2 - (3.0 / 8.0);
            zHat(D < 2 * sqrt(3/8)) = 0;
        case 'exact_unbiased'
            term1 = (D ./ 2.0).^2;
            term2 = (1.0 / 4.0) .* sqrt(3.0 / 2.0) .* (D .^ -1);
            term3 = -(11.0 / 8.0) .* (D .^ -2);
            term4 = (5.0 / 8.0) .* sqrt(3.0 / 2.0) .* (D .^ -3);
            term5 = -(1.0 / 8.0);
            zHat = term1 + term2 + term3 + term4 + term5;
            zHat(D <= 2.0 * sqrt(3.0 / 8.0)) = 0;
        otherwise
            error('Unknown method: %s', method);
    end
    zHat = max(zHat, 0);
end

function claheRGB = enhanceCLAHE_LAB(rgbImg, fovMask, clipLimit, numTiles)
    labImg = rgb2lab(rgbImg);
    L = min(max(labImg(:, :, 1) / 100.0, 0), 1);
    L_clahe = adapthisteq(L, 'NumTiles', numTiles, 'ClipLimit', clipLimit, 'Distribution', 'uniform');
    labImg(:, :, 1) = L_clahe * 100.0;
    claheRGB = lab2rgb(labImg, 'OutputType', 'uint8');
    innerMask = imerode(fovMask, strel('disk', 2));
    claheRGB  = claheRGB .* uint8(repmat(innerMask, [1, 1, 3]));
end

function msrcrRGB = enhanceMSRCR(rgbImg, fovMask, scales, weights, alpha, beta, G, b)
    imgDouble = double(rgbImg) + 1.0;
    [H, W, C] = size(imgDouble);
    channelSum = sum(imgDouble, 3);
    msrcrOut = zeros(H, W, C);
    
    for ch = 1:C
        I_c = imgDouble(:, :, ch);
        R_msr = zeros(H, W);
        for s = 1:numel(scales)
            sigma = scales(s);
            kSize = 2 * ceil(3 * sigma) + 1;
            surround = max(imgaussfilt(I_c, sigma, 'FilterSize', kSize, 'Padding', 'symmetric'), 1.0);
            R_msr = R_msr + weights(s) .* (log(I_c) - log(surround));
        end
        C_i = beta .* (log(alpha .* I_c) - log(channelSum));
        msrcrOut(:, :, ch) = G .* (C_i .* R_msr + b);
    end
    
    msrcrUint8 = zeros(H, W, C, 'uint8');
    innerMask = imerode(fovMask, strel('disk', 3));
    for ch = 1:C
        chanData = msrcrOut(:, :, ch);
        roiVals  = chanData(innerMask);
        if ~isempty(roiVals)
            lowLim  = prctile(roiVals, 1.0);
            highLim = prctile(roiVals, 99.0);
            if highLim > lowLim
                chanNorm = (chanData - lowLim) ./ (highLim - lowLim);
            else
                chanNorm = mat2gray(chanData);
            end
        else
            chanNorm = mat2gray(chanData);
        end
        msrcrUint8(:, :, ch) = im2uint8(min(max(chanNorm, 0), 1));
    end
    msrcrRGB = msrcrUint8 .* uint8(repmat(innerMask, [1, 1, 3]));
end

function q = computeDIPMetrics(refRGB, testRGB, fovMask)
    if isequal(refRGB, testRGB)
        q.PSNR = Inf; q.SSIM = 1.0;
    else
        q.PSNR = psnr(testRGB, refRGB);
        q.SSIM = ssim(testRGB, refRGB);
    end
    testGreen = im2double(testRGB(:, :, 2));
    roiPixels = testGreen(fovMask);
    if isempty(roiPixels), roiPixels = testGreen(:); end
    q.RMS_Contrast = std(roiPixels);
    I_max = prctile(roiPixels, 99.5);
    I_min = prctile(roiPixels, 0.5);
    if (I_max + I_min) > eps
        q.Michelson_Contrast = (I_max - I_min) / (I_max + I_min);
    else
        q.Michelson_Contrast = 0;
    end
    roiUint8 = testRGB(:, :, 2);
    q.Entropy = entropy(roiUint8(fovMask));
end

function plotStage1Comparison(demoData, metricsLog, figOutDir)
    fig = figure('Name', 'Project 29 - Real APTOS Stage 1 Comparison', ...
                 'Position', [100, 100, 1450, 720], 'Color', 'w', 'Visible', 'off');
    t = tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t, sprintf('Explainable MINet-ViT Stage 1: Real APTOS Fundus Preprocessing (%s)', demoData.name), ...
          'FontSize', 14, 'FontWeight', 'bold');
      
    panels = {demoData.cropped, demoData.clahe, demoData.retinex, demoData.vst};
    mKeys  = {'Raw_Cropped', 'CLAHE_LAB', 'MSRCR_Retinex', 'Anscombe_VST_Domain'};
    titles = {'(a) Original Cropped Fundus', '(b) CLAHE Enhanced (L*a*b*)', ...
              '(c) Multi-Scale Retinex (MSRCR)', '(d) Forward Anscombe VST'};
          
    for k = 1:4
        nexttile(k);
        imshow(panels{k});
        mKey = mKeys{k};
        if k == 1
            subStr = sprintf('RMS Contr: %.3f | Entropy: %.2f bits', ...
                metricsLog.(mKey).RMS_Contr(1), metricsLog.(mKey).Entropy(1));
        else
            subStr = sprintf('PSNR: %.1f dB | SSIM: %.3f | Ent: %.2f', ...
                metricsLog.(mKey).PSNR(1), metricsLog.(mKey).SSIM(1), metricsLog.(mKey).Entropy(1));
        end
        title(titles{k}, 'FontSize', 11, 'FontWeight', 'bold');
        xlabel(subStr, 'FontSize', 9.5);
    end
    
    histColors = {[0.2 0.2 0.2], [0.0 0.45 0.74], [0.85 0.33 0.10], [0.47 0.67 0.19]};
    for k = 1:4
        nexttile(k + 4);
        imgG = panels{k}(:, :, 2);
        roiG = imgG(demoData.mask);
        histogram(double(roiG), 0:4:255, 'FaceColor', histColors{k}, 'EdgeColor', 'none', 'Normalization', 'pdf');
        xlim([0, 255]); grid on;
        title(sprintf('Green-Channel FOV Histogram %s', titles{k}(1:3)), 'FontSize', 10);
        xlabel('Pixel Intensity (0-255)', 'FontSize', 9);
        ylabel('Probability Density', 'FontSize', 9);
    end
    outPng = fullfile(figOutDir, 'Stage1_4Panel_Comparison.png');
    exportgraphics(fig, outPng, 'Resolution', 300);
    close(fig);
end

function generateSyntheticFundusDataset(targetDir, numFiles)
    H = 512; W = 512; [X, Y] = meshgrid(1:W, 1:H);
    cx = W/2; cy = H/2; R = 220; fovMask = ((X - cx).^2 + (Y - cy).^2) <= R^2;
    for idx = 1:numFiles
        illum = max(0.35, 1.0 - 0.55 * ((sqrt((X-cx).^2 + (Y-cy).^2)/R).^1.8));
        rgb = uint8(cat(3, 185*illum, 75*illum, 35*illum)) .* uint8(repmat(fovMask, [1,1,3]));
        imwrite(rgb, fullfile(targetDir, sprintf('sample_fundus_%02d.png', idx)));
    end
end