function metric_gui
clc; close all;

%% ========== GUI 窗口 ==========
fig = figure('Name','Video/Image Quality Evaluation Tool (PSNR/SSIM)',...
    'Position',[300 150 900 650]);

handles = struct;
handles.numSR = 3; % 默认3个SR结果
handles.metricType = 'PSNR'; % 默认指标

%% ========== 指标类型选择 ==========
uicontrol(fig,'Style','text',...
    'Position',[30 590 100 25],...
    'String','Metric Type:',...
    'HorizontalAlignment','left');

handles.metricSelector = uicontrol(fig,'Style','popupmenu',...
    'Position',[140 590 100 25],...
    'String',{'PSNR','SSIM'},...
    'Value',1,...
    'Callback',@updateMetricType);

%% ========== SR数量选择 ==========
uicontrol(fig,'Style','text',...
    'Position',[280 590 150 25],...
    'String','Number of SR Results:',...
    'HorizontalAlignment','left');

handles.numSelector = uicontrol(fig,'Style','popupmenu',...
    'Position',[440 590 60 25],...
    'String',{'3','4','5'},...
    'Value',1,...
    'Callback',@updateFileSelectors);

%% ========== 文件选择框容器 ==========
handles.filePanel = uipanel(fig,'Position',[0.03 0.6 0.94 0.3]);

%% ========== 开始按钮 ==========
uicontrol(fig,'Style','pushbutton',...
    'Position',[390 350 120 40],...
    'String','Start',...
    'FontSize',11,...
    'Callback',@startEvaluation);

%% ========== 平均指标显示区域 ==========
handles.avgLabel = uicontrol(fig,'Style','text',...
    'Position',[30 340 120 25],...
    'String','Average PSNR:',...
    'FontWeight','bold',...
    'HorizontalAlignment','left');

handles.avgPanel = uipanel(fig,'Position',[0.03 0.46 0.94 0.06]);

%% ========== 坐标轴 ==========
handles.ax = axes(fig,'Position',[0.08 0.08 0.85 0.35]);
grid on;
xlabel('Frame Index');
ylabel('PSNR (dB)');
title('PSNR Curve');

% 初始化文件和扩展名数组
handles.files = {};
handles.ext = {};

guidata(fig,handles);
updateFileSelectors(); % 初始化文件选择框

%% ================== 回调函数 ==================

    function updateMetricType(~,~)
        handles = guidata(fig);
        
        % 获取选择的指标类型
        selectedIdx = get(handles.metricSelector,'Value');
        metricOptions = get(handles.metricSelector,'String');
        handles.metricType = metricOptions{selectedIdx};
        
        % 更新标签
        if strcmp(handles.metricType, 'PSNR')
            set(handles.avgLabel, 'String', 'Average PSNR:');
            ylabel(handles.ax, 'PSNR (dB)');
            title(handles.ax, 'PSNR Curve');
        else
            set(handles.avgLabel, 'String', 'Average SSIM:');
            ylabel(handles.ax, 'SSIM');
            title(handles.ax, 'SSIM Curve');
        end
        
        % 清空旧结果
        if isfield(handles, 'avgText')
            for i = 1:length(handles.avgText)
                set(handles.avgText(i), 'String', sprintf('SR%d: --', i));
            end
        end
        
        cla(handles.ax);
        
        guidata(fig,handles);
    end

    function updateFileSelectors(~,~)
        handles = guidata(fig);
        
        % 获取选择的SR数量
        selectedIdx = get(handles.numSelector,'Value');
        numOptions = get(handles.numSelector,'String');
        handles.numSR = str2double(numOptions{selectedIdx});
        
        % 清空之前的控件
        delete(get(handles.filePanel,'Children'));
        delete(get(handles.avgPanel,'Children'));
        
        % 清空文件选择记录
        handles.files = {};
        handles.ext = {};
        
        % 创建文件选择框
        totalFiles = handles.numSR + 1; % SR结果 + GT
        labels = cell(1, totalFiles);
        for i = 1:handles.numSR
            labels{i} = sprintf('SR Result %d', i);
        end
        labels{end} = 'Ground Truth';
        
        % 动态计算位置
        panelHeight = 180;
        spacing = panelHeight / (totalFiles + 1);
        
        handles.edit = [];
        for i = 1:totalFiles
            y_pos = panelHeight - i * spacing;
            
            uicontrol(handles.filePanel,'Style','text',...
                'Position',[10 y_pos 100 20],...
                'String',labels{i},'HorizontalAlignment','left');
            
            handles.edit(i) = uicontrol(handles.filePanel,'Style','edit',...
                'Position',[120 y_pos 550 20],...
                'Enable','inactive');
            
            uicontrol(handles.filePanel,'Style','pushbutton',...
                'Position',[680 y_pos 80 20],...
                'String','Browse',...
                'Callback',{@selectFile, i});
        end
        
        % 创建指标显示文本
        colors = {[1 0 0], [0 0.5 0], [0 0 1], [0.8 0.4 0], [0.6 0 0.6]};
        avgWidth = 800 / handles.numSR;
        
        handles.avgText = [];
        for i = 1:handles.numSR
            x_pos = 10 + (i-1) * avgWidth;
            handles.avgText(i) = uicontrol(handles.avgPanel,'Style','text',...
                'Position',[x_pos 5 avgWidth-10 25],...
                'String',sprintf('SR%d: --', i),...
                'ForegroundColor',colors{i},...
                'FontSize',10,...
                'HorizontalAlignment','left');
        end
        
        guidata(fig,handles);
    end

    function selectFile(~, ~, idx)
        [file,path] = uigetfile({'*.mp4;*.avi;*.png;*.jpg;*.bmp;*.jpeg','Video or Image'},...
                                'Select file');
        if file == 0
            return;
        end
        handles = guidata(fig);
        handles.files{idx} = fullfile(path,file);
        [~,~,ext] = fileparts(file);
        handles.ext{idx} = lower(ext);
        set(handles.edit(idx),'String',handles.files{idx});
        guidata(fig,handles);
    end

    function startEvaluation(~,~)
        handles = guidata(fig);
        
        totalFiles = handles.numSR + 1;
        
        if ~isfield(handles,'files') || numel(handles.files) < totalFiles
            errordlg('Please select all files.');
            return;
        end
        
        % 检查文件格式是否一致
        if numel(unique(handles.ext(1:totalFiles))) > 1
            errordlg('All inputs must have the same file format.');
            return;
        end
        
        % 清空旧结果
        for i = 1:handles.numSR
            set(handles.avgText(i),'String',sprintf('SR%d: --', i));
        end
        
        ext = handles.ext{1};
        cla(handles.ax);
        axes(handles.ax);
        hold on;
        
        if contains(ext,{'.mp4','.avi'})
            computeVideoMetrics(handles);
        else
            computeImageMetrics(handles);
        end
        
        grid on;
        hold off;
    end

%% ========== 视频指标计算 ==========
    function computeVideoMetrics(handles)
        numSR = handles.numSR;
        metricType = handles.metricType;
        
        % 打开所有视频
        vr_sr = cell(1, numSR);
        for i = 1:numSR
            vr_sr{i} = VideoReader(handles.files{i});
        end
        vr_gt = VideoReader(handles.files{numSR+1});
        
        idx = 1;
        metric_values = cell(1, numSR);
        for i = 1:numSR
            metric_values{i} = [];
        end
        
        % 检查所有视频是否有帧
        while hasFrame(vr_gt)
            hasAllFrames = true;
            for i = 1:numSR
                if ~hasFrame(vr_sr{i})
                    hasAllFrames = false;
                    break;
                end
            end
            if ~hasAllFrames
                break;
            end
            
            gt = im2double(readFrame(vr_gt));
            
            for i = 1:numSR
                sr = im2double(readFrame(vr_sr{i}));
                
                % 根据选择的指标类型计算
                if strcmp(metricType, 'PSNR')
                    metric_values{i}(idx) = psnr(sr, gt);
                else % SSIM
                    metric_values{i}(idx) = ssim(sr, gt);
                end
            end
            
            idx = idx + 1;
        end
        
        % 绘图
        colors = {'r','g','b',[0.8 0.4 0],[0.6 0 0.6]};
        legendLabels = cell(1, numSR);
        
        for i = 1:numSR
            avg = mean(metric_values{i});
            plot(metric_values{i},'Color',colors{i},'LineWidth',1.5);
            legendLabels{i} = sprintf('SR Method %d', i);
            
            % 显示结果
            if strcmp(metricType, 'PSNR')
                set(handles.avgText(i),'String',sprintf('SR%d: %.4f dB', i, avg));
            else
                set(handles.avgText(i),'String',sprintf('SR%d: %.4f', i, avg));
            end
        end
        
        legend(legendLabels,'Location','best');
        xlabel('Frame Index');
        
        if strcmp(metricType, 'PSNR')
            ylabel('PSNR (dB)');
            title('PSNR Curve');
        else
            ylabel('SSIM');
            title('SSIM Curve');
            ylim([0 1]);
        end
    end

%% ========== 图像指标计算 ==========
    function computeImageMetrics(handles)
        numSR = handles.numSR;
        metricType = handles.metricType;
        
        img_gt = im2double(imread(handles.files{numSR+1}));
        metric_values = zeros(1, numSR);
        
        for i = 1:numSR
            img_sr = im2double(imread(handles.files{i}));
            
            % 根据选择的指标类型计算
            if strcmp(metricType, 'PSNR')
                metric_values(i) = psnr(img_sr, img_gt);
            else % SSIM
                metric_values(i) = ssim(img_sr, img_gt);
            end
        end
        
        % 绘制柱状图
        colors = {[1 0 0], [0 0.5 0], [0 0 1], [0.8 0.4 0], [0.6 0 0.6]};
        b = bar(metric_values);
        b.FaceColor = 'flat';
        for i = 1:numSR
            b.CData(i,:) = colors{i};
        end
        
        xlabels = cell(1, numSR);
        for i = 1:numSR
            xlabels{i} = sprintf('SR%d', i);
        end
        
        set(gca,'XTick',1:numSR);
        set(gca,'XTickLabel',xlabels);
        
        if strcmp(metricType, 'PSNR')
            ylabel('PSNR (dB)');
            title('PSNR Comparison');
        else
            ylabel('SSIM');
            title('SSIM Comparison');
            ylim([0 1]);
        end
        
        % 在柱状图上显示数值
        for i = 1:numSR
            text(i, metric_values(i), sprintf('%.4f', metric_values(i)), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        end
        
        % 显示数值
        for i = 1:numSR
            if strcmp(metricType, 'PSNR')
                set(handles.avgText(i),'String',sprintf('SR%d: %.4f dB', i, metric_values(i)));
            else
                set(handles.avgText(i),'String',sprintf('SR%d: %.4f', i, metric_values(i)));
            end
        end
    end
end