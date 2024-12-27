
%*****1.数据集长度，分为训练和测试0.8：0.2
N_input = length(dataset_s);
N_train = floor(0.8* N_input);
N_test = N_input - N_train;
%*****2.格式转换为元胞数组，1*10201的cell，每个cell为64*64
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
%分为训练和测试0.8：0.2
[imgs, origins] = next_batch(N_input, imgs, origins);
train_imgs = imgs(1:N_train, :, :, :);
train_origins = origins(1:N_train, :, :, :);
test_imgs = imgs(N_train+1:end, :, :, :);
test_origins = origins(N_train+1:end, :, :, :);
%归一化处理
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
    sequenceInputLayer(64*64,'Normalization','none','Name','Input Layer')  % 杈撳叆灞傦紝閫傚簲64x64鐨勫浘鍍?
    fullyConnectedLayer(64*64)  % 闅愯棌灞?
    sigmoidLayer  % sigmoid婵?椿鍑芥暟
    dropoutLayer(0.2)  % Dropout灞?
    fullyConnectedLayer(64*64)  % 杈撳嚭灞傦紝澶у皬閫傚簲杈撳嚭鍥惧儚
    sigmoidLayer  % sigmoid婵?椿鍑芥暟
    regressionLayer% 鍥炲綊灞?
];
% 64*64展开为4096
test_imgs = reshape(test_imgs, [N_test, 64*64,1, 1]);
test_origins = reshape(test_origins, [N_test, 64*64,1, 1]);
inputData = reshape(train_imgs, [N_train, 64*64,1, 1]);
targetData = reshape(train_origins, [N_train, 64*64,1, 1]);
%格式转换为元胞数组，每个cell为4096*1
test_imgs = num2cell(test_imgs,[2,3]);
test_origins = num2cell(test_origins,[2,3]);
inputData = num2cell(inputData,[2,3]);
targetData = num2cell(targetData,[2,3]);
%**********************转置*******************************
for i = 1:numel(inputData)
    inputData{i} = inputData{i}'; % 浣跨敤'鏉ヨ繘琛岃浆缃搷浣?
end
for i = 1:numel(targetData)
    targetData{i} = targetData{i}'; % 浣跨敤'鏉ヨ繘琛岃浆缃搷浣?
end
for i = 1:numel(test_imgs)
    test_imgs{i} = test_imgs{i}'; % 
end
for i = 1:numel(test_origins)
    test_origins{i} = test_origins{i}'; % 浣跨敤'鏉ヨ繘琛岃浆缃搷浣?
end
loss = @ssimLoss; 
% 璁剧疆璁粌閫夐」
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
    );  % 浣跨敤GPU

%训练
%net = trainNetwork(inputData, targetData, layers, options);
disp('训练完成');


% 测试结果
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
    % 添加两条垂直的白色竖线
    % 设置线条的属性（白色竖线）
    % lineColor = 'white';
    % lineWidth = 1; % 线宽度
    % line([32, 32], ylim(ax), 'Color', lineColor, 'LineWidth', lineWidth);
    % line(xlim(ax), [32, 32], 'Color', lineColor, 'LineWidth', lineWidth);
end
% function loss = ssimLoss(y_true, y_pred)
%     % 鍦ㄦ鑷畾涔?SSIM 鎹熷け鍑芥暟鐨勮绠?
%     y_true = reshape(y_true,[64 64]);
%     y_pred = reshape(y_pred,[64 64]);
%     loss = 1-ssim(y_true, y_pred); % 浣犻渶瑕佺紪鍐欎竴涓嚱鏁版潵璁＄畻 SSIM
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
    predictedOutputi = double(reshape(predictedOutputi, [64 64]));%重建图像
    [rectxp,rectyp,areap]  = pre_box(predictedOutputi);%计算矩形中心
    [rectx,recty,area]  = pre_box(test_origin);
    err = ((round((rectx(1)+rectx(3))*0.5)-round((rectxp(1)+rectxp(3))*0.5))^2+...
    (round((recty(1)+recty(3))*0.5)-round((rectyp(1)+rectyp(3))*0.5))^2)^0.5;
    err_mean = err+err_mean;
    err_max = max(err_max, err);
    % 计算各项指标
    %计算ssim指标
    ssim_mean = ssim_mean+ssim(test_origin,predictedOutputi);
    %计算mse指标
    mse_value = sum((test_origin(:) - predictedOutputi(:)).^2) / numel(test_origin);
    mse_mean = mse_mean+mse_value;
    mse_max = max(mse_value,mse_max);
    %计算psnr指标
    
    max_possible_value = double(max(test_origin(:)));
    psnr_mean = psnr_mean + 10 * log10((max_possible_value^2) / mse_value);
    % 计算信息熵差值
    entropy_mean = entropy_mean + abs(entropy(test_origin) - entropy(predictedOutputi));
end
err_mean = err_mean/N_test
ssim_mean = ssim_mean/N_test
mse_mean = mse_mean/N_test
psnr_mean = psnr_mean/N_test
entropy_mean = entropy_mean/N_test