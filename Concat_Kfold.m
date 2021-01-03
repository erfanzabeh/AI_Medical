function [B,B0] = Concat_Kfold(SVMModel,fold_number)
for fold = 1:fold_number
beta(fold,:) = SVMModel.Trained{fold}.Beta;
bias(fold) = SVMModel.Trained{fold}.Bias;
end
B = mean(beta);
B0 = mean(bias);
end