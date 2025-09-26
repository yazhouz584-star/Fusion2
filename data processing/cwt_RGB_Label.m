clc; clear; close all; 
addpath(genpath(pwd));

% 初始化参数
Samples = 50;  % 每类故障生成的样本数
ImageL = 64; ImageW = ImageL; ImageSize = ImageL * ImageW;
dataPoints = ImageSize;
train_ratio = 0.6;
valid_ratio = 0.2;
test_ratio = 0.2;

data_dir = 'F:\Paderborn\';
dataset_dir = "50";

% 创建数据集文件夹
mkdir(dataset_dir);
mkdir(fullfile(dataset_dir, "train"));
mkdir(fullfile(dataset_dir, "valid"));
mkdir(fullfile(dataset_dir, "test"));

% 指定要读取的.mat文件列表
MAT_FILES = ["K001_1.mat", "KA01_1.mat", "KA05_1.mat", "KA07_1.mat", "KI01_1.mat", "KI05_1.mat", "KI07_1.mat"];

tic
for iFile = 1:numel(MAT_FILES)
    mat_name = MAT_FILES(iFile);
    mat_path = fullfile(data_dir, mat_name);
    
    % 加载.mat文件
    S = load(mat_path);
    
    % 获取变量名
    var_names = fieldnames(S);
    mat_variable = S.(var_names{1});
    
    length4Khz = 16000;
    length64Khz = length4Khz * 16;

    signal_current_1 = mat_variable.Y(2).Data(1:length64Khz)';
    signal_current_2 = mat_variable.Y(3).Data(1:length64Khz)';
    signal_vibration = mat_variable.Y(7).Data(1:length64Khz)';

    signal_force = resample(mat_variable.Y(1).Data', 64000, 4000);
    signal_torque = resample(mat_variable.Y(6).Data', 64000, 4000);

    minLength = min([length(signal_current_1), length(signal_current_2), ...
                     length(signal_vibration), length(signal_force), length(signal_torque)]);

    signal_current_1 = signal_current_1(1:minLength);
    signal_current_2 = signal_current_2(1:minLength);
    signal_vibration = signal_vibration(1:minLength);
    signal_force = signal_force(1:minLength);
    signal_torque = signal_torque(1:minLength);

    dataRaw = [signal_current_1, signal_current_2, signal_vibration, signal_force, signal_torque];
    data = pca_noexplained(dataRaw, 3);

    maxIndex = max(0, length(data) - dataPoints - 1);
    randomSerial = round(unifrnd(0, 1, 1, Samples) * maxIndex);

    % 根据文件名生成故障标签
    if startsWith(mat_name, 'K001')
        fault_label = '0';
    elseif startsWith(mat_name, 'KA01')
        fault_label = '1';
    elseif startsWith(mat_name, 'KA05')
        fault_label = '2';
    elseif startsWith(mat_name, 'KA07')
        fault_label = '3';
    elseif startsWith(mat_name, 'KI01')
        fault_label = '4';
    elseif startsWith(mat_name, 'KI05')
        fault_label = '5';
    elseif startsWith(mat_name, 'KI07')
        fault_label = '6';
    end

    for iCut = 1:Samples
        cutIndex = randomSerial(iCut);
        signalCut1 = data(cutIndex+1:cutIndex+dataPoints, 1);
        signalCut2 = data(cutIndex+1:cutIndex+dataPoints, 2);
        signalCut3 = data(cutIndex+1:cutIndex+dataPoints, 3);

        scales = 1:128;
        waveletName = 'morl';
        cwt1 = cwt(signalCut1, scales, waveletName);
        cwt2 = cwt(signalCut2, scales, waveletName);
        cwt3 = cwt(signalCut3, scales, waveletName);

        logCWT1 = log(1 + abs(cwt1));
        logCWT2 = log(1 + abs(cwt2));
        logCWT3 = log(1 + abs(cwt3));

        imgCWT1 = uint8(255 * (logCWT1 - min(logCWT1(:))) / (max(logCWT1(:)) - min(logCWT1(:))));
        imgCWT2 = uint8(255 * (logCWT2 - min(logCWT2(:))) / (max(logCWT2(:)) - min(logCWT2(:))));
        imgCWT3 = uint8(255 * (logCWT3 - min(logCWT3(:))) / (max(logCWT3(:)) - min(logCWT3(:))));

        % 加入 Gamma 变换
        gamma = 0.6;
        imgCWT1 = uint8(255 * (double(imgCWT1) / 255) .^ gamma);
        imgCWT2 = uint8(255 * (double(imgCWT2) / 255) .^ gamma);
        imgCWT3 = uint8(255 * (double(imgCWT3) / 255) .^ gamma);

        % 增强对比度
        imgCWT1 = imadjust(imgCWT1, stretchlim(imgCWT1, [0.02, 0.98]));
        imgCWT2 = imadjust(imgCWT2, stretchlim(imgCWT2, [0.02, 0.98]));
        imgCWT3 = imadjust(imgCWT3, stretchlim(imgCWT3, [0.02, 0.98]));

        imgRGB = cat(3, imgCWT1, imgCWT2, imgCWT3);
        imgRGB = imresize(imgRGB, [ImageL, ImageW]);

        r = rand();
        if r < train_ratio
            save_path = fullfile(dataset_dir, "train", strcat(num2str(iCut), '-', fault_label, '.png'));
        elseif r < train_ratio + valid_ratio
            save_path = fullfile(dataset_dir, "valid", strcat(num2str(iCut), '-', fault_label, '.png'));
        else
            save_path = fullfile(dataset_dir, "test", strcat(num2str(iCut), '-', fault_label, '.png'));
        end

        imwrite(imgRGB, save_path);
    end
end

disp(['数据集生成完成，运行时间: ', num2str(toc), ' s']);
