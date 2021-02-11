%% Version 1.0; future versions will update and shorten the code for better use. 

clc
close all
clear
%-----------------------
%%
dat_impute = orig_dat(:, 2:end);
temp_d = readcell('Additional_Data.xlsx', 'Sheet', 'Sheet3');
v_nams = temp_d(1,2:end);
dat = array2table(dat_impute);
dat.Properties.VariableNames = v_nams;

clear temp_d

dat.AbsNeut = dat.Neutr .* dat.WBC / 100;
dat.AbsLymph = dat.Lymph .* dat.WBC / 100;
dat = removevars(dat, {'Neutr', 'Lymph'});
% Do some shuffling
pat_ind = randperm(height(dat));
dat = dat(pat_ind, :);




cc = cvpartition(table2array(dat(:,1)),'HoldOut', 0.2, 'Stratify', true); % save 20 percent for testing
% Split to have test set
train_dat = dat(cc.training, 2:end);
train_outcm = table2array(dat(cc.training, 1));
test_dat = dat(cc.test, 2:end);
test_outcm = table2array(dat(cc.test, 1));
%% Modelling
tre_templ = templateTree('MaxNumSplits', 5, 'Surrogate', true);

%'OptimizeHyperparameters', {'NumlearningCycles', 'LearnRate',
%'MaxNumSplits', MinLeafSize}
% Auto:  {'Method','NumLearningCycles','LearnRate'} + 'MinLeafSize' if tree



parms = hyperparameters('fitcensemble', train_dat, train_outcm, tre_templ); % Use this to change search range
parms(2).Range = [50 500]; %change search range for number of trees
parms(1).Optimize = false; % Method
parms(4).Optimize = false; % Min Leaf Size
parms(3).Range = [0.1 0.8]; % Learning Rate
parms(5).Range = [1 10]; % maximum number of splits
parms(5).Optimize = true;


mdl = fitcensemble(train_dat, train_outcm,'Learners', tre_templ, 'CategoricalPredictors', [2 9 10 11],... Sex, HTN,DM,Cardio
    'OptimizeHyperparameters', parms,'Method', 'LogitBoost', ...
    'HyperparameterOptimizationOptions',struct('Verbose',1, 'repartition', true,...
     'MaxObjectiveEvaluations', 50,'Leaveout', true)) ;  % returns a ClassificationPartitionModel. this model is trained on cross validated folds and instead of
% using crossval function, you can use k
%{
1) NumLearningCycles: number of predictor units(number of trees etc)
2)LearnRate is the learning rate. to use shrinkage, specify a value between
0 and 1. note that with shrinkage you will need more trees or the algorithm
will underfit.
3) 'CategoricalPredictors' specifies the index of categorical variables in
the table of inputs
%}
    
%% Assess Performance

%-------------kfold loss------------------
kflc = mdl.crossval.kfoldLoss('Mode','cumulative'); % the length of this is same as number of trees. it shows the loss
%as new trees are added
figure;
plot(kflc);
ylabel('10-fold Misclassification rate');
xlabel('Learning cycle');

%----------------------------------------
best_model = mdl.ModelParameters;

[joint_train_labls, joint_train_scores, ~] = mdl.crossval.kfoldPredict; %predictions from kfold on train data
joint_performance_tbl = Performance_table(train_outcm, joint_train_labls); % Performance of joint model

% parameters of the best model

[X_joint,Y_joint,T_joint,AUC_joint, OptPoint_joint] = perfcurve(train_outcm,joint_train_scores(:,1),0); 
% here you should give it the model outcome and scores of the 
% class that you want to inspect as the positive class and also the name or
% value of the positve class
figure
plot(X_joint,Y_joint, '--', 'Color', [1 0.5 0], 'LineWidth', 1);
hold on 
plot(OptPoint_joint(1), OptPoint_joint(2), 'd','color', [1 0.5 0])
hold off
legend({['AUC = ' num2str(AUC_joint)], 'Joint optimal operation point'})
title('Training set ROC curve')

figure
cm = confusionchart(train_outcm,joint_train_labls,...
    'RowSummary', 'row-normalized' ); % get confusion matrix on training data
title('Confusion Matrix of training set')

figure
[test_labls, ~] = mdl.predict(test_dat); % Get scores and model values
confusionchart(test_outcm, test_labls, 'RowSummary', 'row-normalized'); 
title('Confusion matrix of test set')



var_names = train_dat.Properties.VariableNames;

figure
plot(mdl.predictorImportance, '--s', 'Markersize', 10, 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'g',...
    'Color', 'k')
xticks(1:width(train_dat));
xticklabels(var_names)
xtickangle(45)
ax = gca;
ax.Box = 'off';
ax.XLim = [1 length(var_names)]; 

%save2pdf('Weights joint model')

%% Model for non-invasive features
noninv_ind = 1:11;
train_dat_noninv = train_dat(:, noninv_ind); % non invasive train data
train_outcm_noninv = train_outcm;
test_dat_noninv = test_dat(:, noninv_ind); % non invasive test data
test_outcm_noninv = test_outcm;
noninv_dat = dat(:, noninv_ind); % non invasive all data
noninv_outcm = outcm;




parms = hyperparameters('fitcensemble', train_dat_noninv, train_outcm_noninv, tre_templ); % Use this to change search range
parms(2).Range = [50 500]; %change search range for number of trees
parms(1).Optimize = false; % Method
parms(4).Optimize = false; % Mean Leaf Size
parms(3).Range = [0.1 0.8]; % Learning Rate
parms(5).Range = [1 10]; % maximum number of splits
parms(5).Optimize = true;

mdl_noninv = fitcensemble(train_dat_noninv, train_outcm_noninv,'Learners', tre_templ, 'CategoricalPredictors', [2 9 10 11],... Sex, HTN,DM,Cardio
    'OptimizeHyperparameters', parms,'Method', 'LogitBoost', ...
    'HyperparameterOptimizationOptions',struct('Verbose',1,'Kfold', 10, 'repartition', true ...
    , 'MaxObjectiveEvaluations', 60)) ; 
%% Assess performance on non-invasive
%-------------kfold loss------------------
kflc_noninv = mdl_noninv.crossval.kfoldLoss('Mode','cumulative'); % the length of this is same as number of trees. it shows the loss
%as new trees are added
figure;
plot(kflc_noninv);
ylabel('10-fold Misclassification rate');
xlabel('Learning cycle');

%----------------------------------------
[labls_train_noninv, scores_train_noninv,~] = mdl_noninv.crossval.kfoldPredict; % predictions of cross validation on train
%data
[test_labls_noninv, ~] = mdl_noninv.predict(test_dat_noninv); % scores and labels of non-inv test set

[X_noninv,Y_noninv,T_noninv,AUC_noninv, OptPoint_noninv] = perfcurve(train_outcm_noninv,scores_train_noninv(:,1),0);
% performance curve of non-inv train


noninv_performance_tbl = Performance_table(train_outcm_noninv, labls_train_noninv); % Performance of joint model


figure
plot(X_noninv,Y_noninv, '--', 'Color', [1 0.5 0], 'LineWidth', 1);
hold on 
plot(OptPoint_noninv(1), OptPoint_noninv(2), 'gd')
hold off
legend({['AUC = ' num2str(AUC_noninv)], 'Non-invasive optimal operating point'})
title('Non-Invasive Training set ROC curve')

figure
confusionchart(train_outcm_noninv,labls_train_noninv); % get confusion matrix on training data
title('Confusion Matrix of training set')

figure
confusionchart(test_outcm_noninv, test_labls_noninv); 
title('Confusion matrix of test set')


figure
plot(mdl_noninv.predictorImportance, '--s', 'Markersize', 10, 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'g'...
    , 'color', 'k')
xticks(1:width(train_dat_noninv));
xticklabels(var_names(1:length(noninv_ind)))
xtickangle(45)
title('Feature weights of non-invasive features')
ax = gca;
ax.Box = 'off';

%save2pdf('Weights non-inv model')
%% Model for invasive features
inv_ind = 12:width(train_dat);
train_dat_inv = train_dat(:, inv_ind); % non invasive train data
train_outcm_inv = train_outcm;
test_dat_inv = test_dat(:, inv_ind); % non invasive test data
test_outcm_inv = test_outcm;
inv_dat = dat(:, inv_ind); % non invasive all data
inv_outcm = outcm;


%'OptimizeHyperparameters', {'NumlearningCycles', 'LearnRate',
%'MaxNumSplits', MinLeafSize}
% Auto:  {'Method','NumLearningCycles','LearnRate'} + 'MinLeafSize' if tree



parms = hyperparameters('fitcensemble', train_dat_noninv, train_outcm_noninv, tre_templ); % Use this to change search range
parms(2).Range = [50 500]; %change search range for number of trees
parms(1).Optimize = false; % Method
parms(4).Optimize = false; % Mean Leaf Size
parms(3).Range = [0.1 0.8]; % Learning Rate
parms(5).Range = [1 10]; % maximum number of splits
parms(5).Optimize = true;

mdl_inv = fitcensemble(train_dat_inv, train_outcm_inv,'Learners', tre_templ,... 
    'OptimizeHyperparameters', parms,'Method', 'LogitBoost', ...
    'HyperparameterOptimizationOptions',struct('Verbose',1, 'repartition', true,...
    'Kfold', 10, 'MaxObjectiveEvaluations', 60)) ; 
%% Assess performance on invasive
%-------------kfold loss------------------
kflc_inv = mdl_inv.crossval.kfoldLoss('Mode','cumulative'); % the length of this is same as number of trees. it shows the loss
%as new trees are added
figure;
plot(kflc_inv);
ylabel('10-fold Misclassification rate');
xlabel('Learning cycle');
title('Cumulative 10-fold loss of the invasive model')
%----------------------------------------

[labls_train_inv, scores_train_inv,~] = mdl_inv.crossval.kfoldPredict; % scores and labels of inv train set

[test_labls_inv, ~] = mdl_inv.predict(test_dat_inv); % labels of inv test set

[X_inv,Y_inv,T_noninv,AUC_inv,train_OptPoint_inv] = perfcurve(train_outcm_inv,scores_train_inv(:,1),0);
% performance curve of inv train

inv_performance_tbl = Performance_table(train_outcm_inv, labls_train_inv); % Performance of joint model on training set


figure
plot(X_inv,Y_inv, '--', 'Color', [1 0.5 0], 'LineWidth', 1);
hold on 
plot(train_OptPoint_inv(1), train_OptPoint_inv(2),'d', 'color', [0.5 0.5 0.5])
hold off
legend({['Invasive train AUC = ' num2str(AUC_inv)], 'Invasive point of optimal operation'})
title('Invasive Training set ROC curve')

figure
confusionchart(train_outcm_inv,labls_train_inv); % get confusion matrix on training data
title('Confusion Matrix of training set')

figure
confusionchart(test_outcm_inv, test_labls_inv); % Confusion matri of the test set
title('Confusion matrix of test set')


figure %Invasive feature importance
plot(mdl_inv.predictorImportance, '--s', 'Markersize', 10, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'g'...
    , 'color', 'k')
xticks(1:width(train_dat_inv));
xticklabels(var_names(length(noninv_ind)+1:end))
xtickangle(45)
title('Feature weights of invasive features')

