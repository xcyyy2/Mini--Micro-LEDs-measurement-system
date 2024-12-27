
%*****1.���ݼ����ȣ���Ϊѵ���Ͳ���0.8��0.2
N_input = length(dataset_s);
N_train = floor(0.8* N_input);
N_test = N_input - N_train;
%*****2.��ʽת��ΪԪ�����飬1*10201��cell��ÿ��cellΪ64*64
% imgs = cell(1, N_input);
% origins = cell(1, N_input);
% for i = 1:N_input
%     imgs{i} = double(dataset_s(:,:,i));
%     origins{i} = double(ground(:,:,i));
% end
% 
% imgs = cat(3, imgs{:});
% origins = cat(3, origins{:}); % 
% 
% 
imgs = dataset_s;
origins = ground;
%��Ϊѵ���Ͳ���0.8��0.2
[imgs, origins] = next_batch(N_input, imgs, origins);
train_imgs = imgs(1:N_train, :, :, :);
train_origins = origins(1:N_train, :, :, :);
test_imgs = imgs(N_train+1:end, :, :, :);
test_origins = origins(N_train+1:end, :, :, :);
%��һ������
for i = 1:N_train
    train_imgs(i, :, :, :) = train_imgs(i, :, :, :) / 256.0;
    train_origins(i, :, :, :) = train_origins(i, :, :, :) / 256.0;
end
for i = 1:N_test
    test_imgs(i, :, :, :) = test_imgs(i, :, :, :) / 256.0;
    test_origins(i, :, :, :) = test_origins(i, :, :, :) / 256.0;
end
figure;imshow(reshape(train_origins(50,:,:),[64,64]),[]);
figure;imshow(reshape(train_imgs(50,:,:),[64,64]),[]);
train_imgs = reshape(train_imgs,[N_train,64,64,1]);
train_origins = reshape(train_origins,[N_train,64,64,1]);
%**************************Set up Neural network*******************************
layers = [
    sequenceInputLayer(64*64,'Normalization','none','Name','Input Layer')  % 输入层，适应64x64的图�?
    fullyConnectedLayer(64*64)  % 隐藏�?
    sigmoidLayer  % sigmoid�?��函数
    dropoutLayer(0.2)  % Dropout�?
    fullyConnectedLayer(64*64)  % 输出层，大小适应输出图像
    sigmoidLayer  % sigmoid�?��函数
    regressionLayer% 回归�?
];
% 64*64չ��Ϊ4096
test_imgs = reshape(test_imgs, [N_test, 64*64,1, 1]);
test_origins = reshape(test_origins, [N_test, 64*64,1, 1]);
inputData = reshape(train_imgs, [N_train, 64*64,1, 1]);
targetData = reshape(train_origins, [N_train, 64*64,1, 1]);
%��ʽת��ΪԪ�����飬ÿ��cellΪ4096*1
test_imgs = num2cell(test_imgs,[2,3]);
test_origins = num2cell(test_origins,[2,3]);
inputData = num2cell(inputData,[2,3]);
targetData = num2cell(targetData,[2,3]);
%**********************ת��*******************************
for i = 1:numel(inputData)
    inputData{i} = inputData{i}'; % 使用'来进行转置操�?
end
for i = 1:numel(targetData)
    targetData{i} = targetData{i}'; % 使用'来进行转置操�?
end
for i = 1:numel(test_imgs)
    test_imgs{i} = test_imgs{i}'; % 
end
for i = 1:numel(test_origins)
    test_origins{i} = test_origins{i}'; % 使用'来进行转置操�?
end
loss = @ssimLoss; 
% 设置训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', 7000, ...
    'InitialLearnRate', 0.0001, ...
    'MiniBatchSize', 256, ...
    'Plots', 'training-progress', ...
    'ValidationData',{test_imgs,test_origins}, ...
    'ValidationFrequency',10, ...
    'OutputNetwork', 'best-validation-loss', ...
    'Verbose',false, ...
    'ExecutionEnvironment', 'gpu' ...
    );  % 使用GPU

%ѵ��
%net = trainNetwork(inputData, targetData, layers, options);
disp('ѵ�����');


% ���Խ��
figure;
for i = 1:5
    idx = floor(rand*N_test);
    test_img = test_imgs{idx};
    test_origin = test_origins{idx};
    predictedOutputi = predict(net, test_img);
    subplot(3,5,i)
    test_img = reshape(test_img, [64 64]);
    imshow(test_img,[])
    subplot(3,5,i+5)
    test_origin = reshape(test_origin, [64 64]);
    imshow(test_origin,[])
    subplot(3,5,i+10)
    predictedOutputi = double(reshape(predictedOutputi, [64 64]));
    imshow(predictedOutputi,[]);hold on
    % [rectx,recty,area]  = pre_box(predictedOutputi);
    % line(rectx(:),recty(:),'color','b','Linewidth',2);
    % scatter(round((rectx(1)+rectx(3))*0.5),round((recty(1)+recty(3))*0.5),10,'b','filled');
    % ax = gca;
    % ���������ֱ�İ�ɫ����
    % �������������ԣ���ɫ���ߣ�
    % lineColor = 'white';
    % lineWidth = 1; % �߿��
    % line([32, 32], ylim(ax), 'Color', lineColor, 'LineWidth', lineWidth);
    % line(xlim(ax), [32, 32], 'Color', lineColor, 'LineWidth', lineWidth);
end
% function loss = ssimLoss(y_true, y_pred)
%     % 在此自定�?SSIM 损失函数的计�?
%     y_true = reshape(y_true,[64 64]);
%     y_pred = reshape(y_pred,[64 64]);
%     loss = 1-ssim(y_true, y_pred); % 你需要编写一个函数来计算 SSIM
% end
err_mean = 0;
err_max = 0;
mse_max = 0;
ssim_mean = 0;
mse_mean = 0;
psnr_mean = 0;
entropy_mean = 0;
for i = 1:N_test
    test_img = test_imgs{i};
    test_origin = test_origins{i};
    test_origin = reshape(test_origin, [64 64]);
    predictedOutputi = predict(net, test_img);
    predictedOutputi = double(reshape(predictedOutputi, [64 64]));%�ؽ�ͼ��
    [rectxp,rectyp,areap]  = pre_box(predictedOutputi);%�����������
    [rectx,recty,area]  = pre_box(test_origin);
    err = ((round((rectx(1)+rectx(3))*0.5)-round((rectxp(1)+rectxp(3))*0.5))^2+...
    (round((recty(1)+recty(3))*0.5)-round((rectyp(1)+rectyp(3))*0.5))^2)^0.5;
    err_mean = err+err_mean;
    err_max = max(err_max, err);
    % �������ָ��
    %����ssimָ��
    ssim_mean = ssim_mean+ssim(test_origin,predictedOutputi);
    %����mseָ��
    mse_value = sum((test_origin(:) - predictedOutputi(:)).^2) / numel(test_origin);
    mse_mean = mse_mean+mse_value;
    mse_max = max(mse_value,mse_max);
    %����psnrָ��
    
    max_possible_value = double(max(test_origin(:)));
    psnr_mean = psnr_mean + 10 * log10((max_possible_value^2) / mse_value);
    % ������Ϣ�ز�ֵ
    entropy_mean = entropy_mean + abs(entropy(test_origin) - entropy(predictedOutputi));
end
err_mean = err_mean/N_test
ssim_mean = ssim_mean/N_test
mse_mean = mse_mean/N_test
psnr_mean = psnr_mean/N_test
entropy_mean = entropy_mean/N_test