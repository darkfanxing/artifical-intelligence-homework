function accuracy = kNN(k, featureCrosses, rawData, label)
    trainInput = [
                    rawData(1:25, featureCrosses);
                    rawData(51:75, featureCrosses);
                    rawData(101:125, featureCrosses);
                 ]; 
             
    trainLabel = [
                    label(1:25);
                    label(51:75);
                    label(101:125);
                 ];

    testInput = [
                    rawData(26:50, featureCrosses);
                    rawData(76:100, featureCrosses);
                    rawData(126:150, featureCrosses);
                ];
            
    testLabel = [
                    label(26:50);
                    label(76:100);
                    label(126:150);
                ];
    
    RightJudgementCount = 0;
    for testCount = 1:75
        distanceList = [];
        
        for trainCount = 1:75
            distance = norm(testInput(testCount, :) - trainInput(trainCount, :));
            distanceList = [distanceList, distance];
        end
        
        minDistanceList = mink(distanceList, k);
        outputs = [];
        for index = 1:length(minDistanceList)
            minDistance = find(distanceList == minDistanceList(index));
            outputs = [outputs, minDistance];
        end
        
        prediction = mode(trainLabel(outputs));         
        
        if prediction == testLabel([testCount])
            RightJudgementCount = RightJudgementCount + 1;
        end
        
    end
    
    accuracy = RightJudgementCount / 75;
end