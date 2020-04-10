%% Artificial Intelligence Homework #1 - 2020/03/17

%%
clc;
clear;
close all;

%%
dataSet = load('../dataset/iris.txt');
rawData = dataSet(:, 1:4);
label = dataSet(:, 5);
%% Scatter Plot

for i = 1:4
    for j = i:4
        if i == j
            continue;
        end
        
        figure;
        
        plot(rawData(1:50, i), rawData(1:50, j), 'ro',...
             rawData(51:100, i), rawData(51:100, j), 'go',...
             rawData(101:150, i), rawData(101:150, j), 'bo');
        
        title('Scatter Plot');
        legend('class1', 'class2', 'class3');
        xlabel(['Feature' num2str(i)]);
        ylabel(['Feature' num2str(j)]);
    end
end

%% k-NN model

% get 15 feature crosses
FeatureCombination = getFeatureCombination();

kList = {1, 3};

for kIndex = 1:length(kList)
    for index = 1:length(FeatureCrosses)
        accuracy = kNN(kList{kIndex}, FeatureCrosses{index}, rawData, label);
        disp(accuracy);
    end
end

function featureCrossesList = getFeatureCrosses()
    featureCrossesList = {};
    featureCrossesList(end + 1) = {[1, 2, 3, 4]};
    
    for i = 1:4
        % 1, 2 ,3 ,4
        featureCrossesList(end + 1) = {[i]};
        for j = i:4
            % [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]
            if i == j
                continue;    
            end

            featureCrossesList(end + 1) = {[i, j]};
        end
    end

    for i = 1:2
        for j = i + 1:3
            for k = j + 1:4
                % [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]
                featureCrossesList(end + 1) = {[i, j, k]};
            end
        end
    end
end