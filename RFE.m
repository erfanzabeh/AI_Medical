%% Version 1.0; future versions will update and shorten the code for better use. 
% Recursive feature elimination using linear SVM


orig_dat = readmatrix('Additional_Data.xlsx', 'Sheet', 'Sheet3'); % some data features are returnd as cells using readtable, use this trick to get

dat_im = orig_dat(:, 2:end);
temp_d = readcell('Additional_Data.xlsx', 'Sheet', 'Sheet3');
v_nams = temp_d(1,2:end);
dat = array2table(dat_im);
dat.Properties.VariableNames = v_nams;

clear temp_d

dat.AbsNeut = dat.Neutr .* dat.WBC / 100;
dat.AbsLymph = dat.Lymph .* dat.WBC / 100;
dat = removevars(dat, {'Neutr', 'Lymph'});


dat = dat(sum(ismissing(dat),2) == 0, :);


dat_mat = dat(:, 2:end); %convert the input data to matrix format
%dat_cat = dat_mat(:, [2 9 10 11]);
%dat_mat(:, [2 9 10 11]) = [];
%dat_mat = [dat_mat dat_cat];

%---


var_nam = dat_mat.Properties.VariableNames;
var_nam = var_nam(1:end);

dat_mat = table2array(dat_mat);
dat_mat = zscore(dat_mat);

logic_outcm = logical(table2array(dat(:, 1))); % convert the outcomes into logical format;
%%
 Leave1_joint_mdl = fitcsvm(dat_mat, logic_outcm, 'KernelFunction','linear', 'Solver' , 'L1QP',...
    'verbose', 0, 'ClassNames', [1 0],'Leaveout', 'on',...
    'PredictorNames',var_nam, 'verbose',0);

 
%% Plot lambda vs AUC and show the number of features
orig_noninv_datmat = dat_mat(:, 1:11);
noninvname = var_nam(1:11);

repeat_num = 50;

noninv_RFE_Accuracy = nan(repeat_num, length(noninvname)) ;
noninv_RFE_Loss = nan(repeat_num, length(noninvname));
noninv_RFE_AUC = nan(repeat_num, length(noninvname));

noninv_droped_nam = cell(repeat_num, length(noninvname));

tic

for repeat = 1:repeat_num
    for nonfeati = 1:length(noninvname)
        disp(['repeat ' num2str(repeat) ' iter ' num2str(nonfeati)])
        if nonfeati==1
            noninv_datmat =orig_noninv_datmat;
            temp_noninvnam = noninvname;
        end
        
        noninv_mdl = fitcsvm(noninv_datmat, logic_outcm, 'KernelFunction','linear', 'Solver' , 'L1QP',...
            'verbose', 0, 'ClassNames', [1 0],...
            'PredictorNames',temp_noninvnam, 'OptimizeHyperparameters', 'KernelScale',...
            'HyperparameterOptimizationOptions', struct('KFold', 10, 'verbose', 0, 'MaxObjectiveEvaluation', 10,...
            'ShowPlots', false),...
            'PredictorNames', temp_noninvnam );
        
        [noninv_labls_temp, noninv_scores_temp] = noninv_mdl.crossval.kfoldPredict; %labels and scores
        
        noninv_RFE_Loss(repeat, nonfeati) = noninv_mdl.crossval.kfoldLoss;%Loss of iteration
        
        [~,~,~, temp_AUC] = perfcurve(logic_outcm, noninv_scores_temp(:,2), 0); %AUC
        noninv_RFE_AUC(repeat, nonfeati) = temp_AUC;
        
        TPnoninv = sum((logic_outcm == 0) & (noninv_labls_temp == 0));
        TNnoninv = sum((logic_outcm == 1) & (noninv_labls_temp == 1));
        noninv_RFE_Accuracy(repeat, nonfeati) = (TPnoninv + TNnoninv)/(length(logic_outcm)); %accuracy
        %-----Remove feature with the least beta from the data-------
        
        [~, ind] = min(abs(noninv_mdl.Beta)); %find the minimum beta index and value
        
        noninv_droped_nam(repeat, nonfeati) = noninv_mdl.PredictorNames(ind); %add the name of droped coef
        temp_noninvnam(ind) = [];  %Remove the feature name from predictor names
        noninv_datmat(:, ind) = []; %remove the value of droped coef
        
        
        
        
        
    end
end
toc


%%
orig_inv_datmat = dat_mat(:, 12:end);
invname = var_nam(12:end);

repeat_num = 50;

inv_RFE_Accuracy = nan(repeat_num, length(invname)) ;
inv_RFE_Loss = nan(repeat_num, length(invname));
inv_RFE_AUC = nan(repeat_num, length(invname));

inv_droped_nam = cell(repeat_num, length(invname));

tic

for repeat = 1:repeat_num
    for nonfeati = 1:length(invname)
        disp(['repeat ' num2str(repeat) ' iter ' num2str(nonfeati)])
        if nonfeati==1
            inv_datmat =orig_inv_datmat;
            temp_invnam = invname;
        end
        close all
        inv_mdl = fitcsvm(inv_datmat, logic_outcm, 'KernelFunction','linear', 'Solver' , 'L1QP',...
            'verbose', 0, 'ClassNames', [1 0],...
            'PredictorNames',temp_invnam, 'OptimizeHyperparameters', 'KernelScale',...
            'HyperparameterOptimizationOptions', struct('KFold', 10, 'verbose', 0, 'MaxObjectiveEvaluation', 10,...
            'ShowPlots', false));
        
        [inv_labls_temp, inv_scores_temp] = inv_mdl.crossval.kfoldPredict; %labels and scores
        
        inv_RFE_Loss(repeat, nonfeati) = inv_mdl.crossval.kfoldLoss;%Loss of iteration
        
        [~,~,~, temp_AUC] = perfcurve(logic_outcm, inv_scores_temp(:,2), 0); %AUC
        inv_RFE_AUC(repeat, nonfeati) = temp_AUC;
        
        TPinv = sum((logic_outcm == 0) & (inv_labls_temp == 0));
        TNinv = sum((logic_outcm == 1) & (inv_labls_temp == 1));
        inv_RFE_Accuracy(repeat, nonfeati) = (TPinv + TNinv)/(length(logic_outcm)); %accuracy
        %-----Remove feature with the least beta from the data-------
        
        [~, ind] = min(abs(inv_mdl.Beta)); %find the minimum beta index and value
        
        inv_droped_nam(repeat, nonfeati) = inv_mdl.PredictorNames(ind); %add the name of droped coef
        temp_invnam(ind) = [];  %Remove the feature name from predictor names
        inv_datmat(:, ind) = []; %remove the value of droped coef
        
        
        
        
        
    end
end
toc

%% Plotting
%---------Std of accuracy, loss, auc

noninv_StdAccur = std(noninv_RFE_Accuracy, [], 1);
noninv_SemAccur = noninv_StdAccur(end:-1:1)/sqrt(size(noninv_RFE_Accuracy,1)); %Standard erro of mean(SEM), 
%For plotting, flip them
inv_StdAccur = std(inv_RFE_Accuracy, [], 1);
inv_SemAccur = inv_StdAccur(end:-1:1)/sqrt(size(inv_RFE_Accuracy,1));

noninv_StdLoss = std(noninv_RFE_Loss, [], 1);
noninv_SemdLoss = noninv_StdLoss(end:-1:1)/sqrt(size(noninv_RFE_Accuracy,1));
inv_StdLoss = std(inv_RFE_Loss, [], 1);
inv_SemLoss = inv_StdLoss(end:-1:1)/sqrt(size(inv_RFE_Accuracy,1));

noninv_StdAUC = std(noninv_RFE_AUC, [], 1);
noninv_SemAUC = noninv_StdAUC(end:-1:1)/sqrt(size(noninv_RFE_Accuracy,1));
inv_StdAUC = std(inv_RFE_AUC, [], 1);
inv_SemAUC =inv_StdAUC(end:-1:1)/sqrt(size(inv_RFE_Accuracy,1));

%-----Mean Accur and coast-----
noninv_MeanAccur = mean(noninv_RFE_Accuracy, 1);
noninv_MeanAccur = noninv_MeanAccur(end:-1:1);
inv_MeanAccur = mean(inv_RFE_Accuracy, 1);
inv_MeanAccur = inv_MeanAccur(end:-1:1);

noninv_MeanLoss = mean(noninv_RFE_Loss, 1);
noninv_MeanLoss = noninv_MeanLoss(end:-1:1);
inv_MeanLos = mean(inv_RFE_Loss, 1);
inv_MeanLos = inv_MeanLos (end:-1:1);

noninv_MeanAUC = mean(noninv_RFE_AUC, 1);
noninv_MeanAUC = noninv_MeanAUC(end:-1:1);
inv_MeanAUC = mean(inv_RFE_AUC, 1);
inv_MeanAUC = inv_MeanAUC(end:-1:1);



%=--------------With mean + SEM-------------
figure
hold on
patch([1:11 11:-1:1], [noninv_MeanAccur(1:end)-noninv_StdAccur(1:end) noninv_MeanAccur(end:-1:1)+noninv_StdAccur(end:-1:1)],...
    [0 0.5 0], 'FaceAlpha', 0.2, 'EdgeColor', [0 0.5 0])
patch([1:11 11:-1:1], [noninv_MeanAccur(1:end)-noninv_StdAccur(1:end) noninv_MeanAccur(end:-1:1)+noninv_StdAccur(end:-1:1)],...
    [0 0.5 0], 'FaceAlpha', 0.2, 'EdgeColor', [0 0.5 0])
plot(1:11, noninv_MeanAccur(1:end), '--o', 'Markeredgecolor', 'k',...
    'markerfacecolor', [0 0.5 0], 'markersize', 6, 'color', 'k');
 
patch([1:26 26:-1:1], [inv_MeanAccur(1:end)-inv_StdAccur(1:end) inv_MeanAccur(end:-1:1)+inv_StdAccur(end:-1:1)],...
    'w','FaceColor', [0.7 0.7 0.7], 'FaceAlpha', 0.9,'EdgeColor', [0.5 0.5 0.5])
patch([1:26 26:-1:1], [inv_MeanAccur(1:end)-inv_StdAccur(1:end) inv_MeanAccur(end:-1:1)+inv_StdAccur(end:-1:1)],...
    'w','FaceColor', [0.7 0.7 0.7], 'FaceAlpha', 0.9,'EdgeColor', [0.5 0.5 0.5])
plot(1:26, inv_MeanAccur(1:end), '--o', 'Markeredgecolor', 'k',...
    'markerfacecolor', [0.4 0.4 0.4], 'markersize', 6, 'Color', 'k' );

h1= plot([3 3], get(gca, 'ylim'), '--', 'color', [0 0.5 0] )
h2 = plot([6 6], get(gca, 'ylim'), '--', 'color', [0.4 0.4 0.4])
legend([h1 h2],'Non-invasive optimal feature number = 3','invasive optimal feature number = 6',...
    'Location', 'southeast', 'Box', 'off')
grid on
xlim([0 27])


ax = gca;
ax.Box = 'off'
xlabel('Number of features')
ylabel('Accuracy')
axis square
hold off
% save2pdf('Figure3bRFE+SEM')

%---------------With AUC-------------
figure
hold on
patch([1:11 11:-1:1], [noninv_MeanAUC(1:end)-noninv_SemAUC(1:end) noninv_MeanAUC(end:-1:1)+noninv_StdAUC(end:-1:1)],...
    [0 0.5 0], 'FaceAlpha', 0.2, 'EdgeColor', [0 0.5 0])
patch([1:11 11:-1:1], [noninv_MeanAUC(1:end)-noninv_StdAUC(1:end) noninv_MeanAUC(end:-1:1)+noninv_StdAUC(end:-1:1)],...
    [0 0.5 0], 'FaceAlpha', 0.2, 'EdgeColor', [0 0.5 0])
plot(1:11, noninv_MeanAUC(1:end), '--o', 'Markeredgecolor', 'k',...
    'markerfacecolor', [0 0.5 0], 'markersize', 6, 'color', 'k');
 
patch([1:26 26:-1:1], [inv_MeanAUC(1:end)-inv_StdAUC(1:end) inv_MeanAUC(end:-1:1)+inv_StdAUC(end:-1:1)],...
    'w','FaceColor', [0.7 0.7 0.7], 'FaceAlpha', 0.9,'EdgeColor', [0.5 0.5 0.5])
patch([1:26 26:-1:1], [inv_MeanAUC(1:end)-inv_StdAUC(1:end) inv_MeanAUC(end:-1:1)+inv_StdAUC(end:-1:1)],...
    'w','FaceColor', [0.7 0.7 0.7], 'FaceAlpha', 0.9,'EdgeColor', [0.5 0.5 0.5])
plot(1:26, inv_MeanAUC(1:end), '--o', 'Markeredgecolor', 'k',...
    'markerfacecolor', [0.4 0.4 0.4], 'markersize', 6, 'Color', 'k' );

h1= plot([3 3], get(gca, 'ylim'), '--', 'color', [0 0.5 0] );
h2 = plot([6 6], get(gca, 'ylim'), '--', 'color', [0.4 0.4 0.4]);
legend([h1 h2],'Non-invasive optimal feature number = 3','invasive optimal feature number = 6',...
    'Location', 'southeast', 'Box', 'off')
grid on
xlim([0 27])
ylim([0.63 0.85])

ax = gca;
ax.Box = 'off'
xlabel('Number of features')
ylabel('Accuracy')
axis square
hold off

%--------------------
figure
hold on
patch([1:11 11:-1:1], [noninv_MeanLoss(1:end)-noninv_StdLoss(1:end) noninv_MeanLoss(end:-1:1)+noninv_StdLoss(end:-1:1)],...
    [0 0.5 0], 'FaceAlpha', 0.2, 'EdgeColor', [0 0.5 0])
patch([1:11 11:-1:1], [noninv_MeanLoss(1:end)-noninv_StdLoss(1:end) noninv_MeanLoss(end:-1:1)+noninv_StdLoss(end:-1:1)],...
    [0 0.5 0], 'FaceAlpha', 0.2, 'EdgeColor', [0 0.5 0])
plot(1:11, noninv_MeanLoss(1:end), '--o', 'Markeredgecolor', 'k',...
    'markerfacecolor', [0 0.5 0], 'markersize', 6, 'color', 'k');
 
patch([1:26 26:-1:1], [inv_MeanLos(1:end)-inv_StdLoss(1:end) inv_MeanLos(end:-1:1)+inv_StdLoss(end:-1:1)],...
    'w','FaceColor', [0.7 0.7 0.7], 'FaceAlpha', 0.9,'EdgeColor', [0.5 0.5 0.5])
patch([1:26 26:-1:1], [inv_MeanLos(1:end)-inv_StdLoss(1:end) inv_MeanLos(end:-1:1)+inv_StdLoss(end:-1:1)],...
    'w','FaceColor', [0.7 0.7 0.7], 'FaceAlpha', 0.9,'EdgeColor', [0.5 0.5 0.5])
plot(1:26, inv_MeanLos(1:end), '--o', 'Markeredgecolor', 'k',...
    'markerfacecolor', [0.4 0.4 0.4], 'markersize', 6, 'Color', 'k' );

h1= plot([3 3], get(gca, 'ylim'), '--', 'color', [0 0.5 0] );
h2 = plot([6 6], get(gca, 'ylim'), '--', 'color', [0.4 0.4 0.4]);
legend([h1 h2],'Non-invasive optimal feature number = 3','invasive optimal feature number = 6',...
    'Location', 'southeast', 'Box', 'off')
grid on
xlim([0 27])
% ylim([0.63 0.85])

ax = gca;
ax.Box = 'off'
xlabel('Number of features')
ylabel('Accuracy')
axis square
hold off






