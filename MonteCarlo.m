%芯片尺寸
height = 40;
width = 80;
%重建的芯片形貌（DNN获得）
img_weight = predictedOutputi;
%小球半径1cm，即5000pixcel
R = 5000;
%已知已插光纤位置的alpha、beta
load('cir_all.mat')
%加载601个光纤对应的光谱，以及权重
load('dataCir.mat')
sumOfData = sum(dataCirSort,2);
counts = sumOfData/max(sumOfData);
%最大光线发射数
maxNumRays = 1000000;
tic
%结果为芯片的高光谱数据，大小101*51*328
resultLED = zeros(40,80,328);%芯片分辨率为1个pixel 2微米
%外推结果
resultPlat_outer = zeros(2001,4001);%外平面分辨率为1个pixel 20
%内推结果
resultPlat_inner = zeros(1001,2001);%内平面分辨率为1个pixel 20
numQuan = [1,12,17,26,32,32,37];%每圈的个数
alphaInterval = 360./numQuan;
for i = 1:601%遍历所有光纤
    %每根光纤发射的光线数量
    numRays = round(counts(i)*maxNumRays);
    %每根光纤的光谱
    specFiber = dataCirSort(i,:);
    %将每根光纤的光谱平均到每条光线
    specRays = specFiber/numRays;
    %将每根光纤代表的区域划分成numRays个小区域，每个小区域光谱为specRays
    alpha = cir_all(i,4);
    beta = cir_all(i,3);
    betaIdx = (beta+5)/5;%该光纤位于第betaIdx圈
    if (betaIdx > 7)
        betaIdx = 7;
    end %从第七圈开始都是37个光纤
    alphaIntervalQuan = alphaInterval(betaIdx);%该光纤的alpha间隔
    %根据alpha和beta间隔以及numRays划分
    areaSqrt = floor(sqrt(numRays));%可以均分的区域
    areaSqrtAlpha = alpha-alphaIntervalQuan:2*alphaIntervalQuan/areaSqrt:alpha+alphaIntervalQuan;
    %光纤的beta间隔固定为5或2.5（顶部底部）
    if (beta == 0)
        areaSqrtbeta = 0:2.5/areaSqrt:2.5;
    elseif (beta == 90)
        areaSqrtbeta = 87.5:2.5/areaSqrt:90;
    else
        areaSqrtbeta = beta-2.5:5/areaSqrt:beta+2.5;
    end
    areaRamdon = numRays - areaSqrt^2;%剩下的区域随机生成
    for j = 1:numRays
        %获取所分区域的索引以及角度
        if (j <= areaSqrt^2)
            areaAlphaIdx = mod(j,areaSqrt);
            if areaAlphaIdx == 0
                areaAlphaIdx = areaSqrt;
            end
            areaBetaIdx = ceil(j/areaSqrt);
            alphaArea = (areaSqrtAlpha(areaAlphaIdx)+areaSqrtAlpha(areaAlphaIdx+1))/2;
            betaArea = (areaSqrtbeta(areaBetaIdx)+areaSqrtbeta(areaBetaIdx+1))/2;
        else
            %剩余区域随机生成
            alphaArea = areaSqrtAlpha(1)+rand*(areaSqrtAlpha(end)-areaSqrtAlpha(1));
            betaArea = areaSqrtbeta(1)+rand*(areaSqrtbeta(end)-areaSqrtbeta(1));
        end
        %根据光纤位置alpha、beta转换为直角坐标(光线起点坐标)
        xArea = R*sin(betaArea/180*pi)*cos(alphaArea/180*pi);
        yArea = R*sin(betaArea/180*pi)*sin(alphaArea/180*pi);
        zArea = R*cos(betaArea/180*pi);
        %单位球内随机产生一点
        thetaLED = rand*pi;
        phiLED = rand*2*pi;
        xLED = xArea + sin(thetaLED) * cos(phiLED);
        yLED = yArea + sin(thetaLED) * sin(phiLED);
        zLED = zArea + cos(thetaLED);
        %根据两点确定直线的参数方程
        t = -2*R:2*R;
        xLine = xArea + t * (xLED - xArea);
        yLine = yArea + t * (yLED - yArea);
        zLine = zArea + t * (zLED - zArea);
        %判断获取直线与所求面的交点
        %1.直线与芯片的交点
        for tLine = 1:numel(t)-1
            %内推
            if(zLine(tLine)*zLine(tLine+1)<=0)%表明过z=0平面
                %获取z=0平面交点
                if(xLine(tLine)>=-20 && xLine(tLine)<=19 && yLine(tLine)>=-39 && yLine(tLine)<=40) || (xLine(tLine+1)>=-20 && xLine(tLine+1)<=19 && yLine(tLine+1)>=-39 && yLine(tLine+1)<=40)
                    %此时交点位于芯片内，计算与x=0平面的交点
                    xLED = xLine(tLine);
                    yLED = yLine(tLine);
                    zLED = 0;
                    %与x=0平面有交点
                    if (xLED > 0 && xArea < 0) || (xLED < 0 && xArea > 0)
                        % 计算X=0时的Y和Z
                        % 线性插值公式
                        y_at_x0 = yLED + (yArea - yLED) * (0 - xLED) / (xArea - xLED);
                        z_at_x0 = zLED + (zArea - zLED) * (0 - xLED) / (xArea - xLED);
                        %交点位于内切面
                        if (y_at_x0^2 + z_at_x0^2) <= 5000^2
                            zPlatIdx = round(z_at_x0/10)+1;
                            yPlatIdx = round(y_at_x0/10)+1001;
                            %根据重建结果加上权重
                            %坐标转换
                            xLEDIdx = 20-round(xLine(tLine));
                            yLEDIdx = round(yLine(tLine))+40;
                            weight = img_weight(xLEDIdx, yLEDIdx);
                            resultPlat_inner(zPlatIdx,yPlatIdx) = resultPlat_inner(zPlatIdx,yPlatIdx) + sum(specRays)*weight;
                        end
                    end
                end 
            end
            %外推
            if(xLine(tLine)*xLine(tLine+1)<=0)%表明过x=0平面
                if (zLine(tLine)>=0 && (zLine(tLine)^2 + yLine(tLine)^2)>=5000^2) || (zLine(tLine+1)>=0 && (zLine(tLine+1)^2 + yLine(tLine+1)^2)>=5000^2)
                    zPlatIdx = round(zLine(tLine)/10)+1;
                    yPlatIdx = round(yLine(tLine)/10)+2001;
                    resultPlat_outer(zPlatIdx,yPlatIdx) = resultPlat_outer(zPlatIdx,yPlatIdx) + sum(specRays);
                end
            end
        end    
    end
    disp(['已模拟光线数' num2str(i)])
end
toc
%figure;imshow(sum(resultLED(2:50,2:100,:),3),[]);colormap('jet');
figure;imshow(resultPlat(:,:),[]);colormap('jet');
% resultPlat_show = resultPlat;
% for i = 1:2001
%     for j = 1:4001
%         if ((i-1)^2+(j-2001)^2)<1000^2
%             resultPlat_show(i,j)=0;
%         end
%     end
% end
% figure;imshow(resultPlat_show(:,:),[]);colormap('jet');
