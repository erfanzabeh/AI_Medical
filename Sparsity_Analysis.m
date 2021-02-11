%% --------------- Sparsity analysis ----------------------
%----------------- Mahdavi et.al 2020---------------------
% Version 1.0; future versions will update and trim the code for better use. 

%% Data extraction and variable initiation

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

%% Plot lambda vs AUC and show the number of features
noninv_datmat = dat_mat(:, 1:11);
non_invname = var_nam(1:11);
inv_datmat = dat_mat(:, 12:end);
invnam = var_nam(12:end);

lambds = logspace(-1.7,-0.5,25);

repeat_num = 100;

%-------------Initialize--------
noninv_size = nan(repeat_num, length(lambds));
inv_size = nan(repeat_num, length(lambds));
inv_cost = nan(repeat_num, length(lambds));
noninv_cost = nan(repeat_num, length(lambds));

inv_accuracy = nan(repeat_num, length(lambds));
noninv_accuracy = nan(repeat_num, length(lambds));
inv_AUC = nan(repeat_num, length(lambds));
noninv_AUC = nan(repeat_num, length(lambds));



noninv_MinLoss = nan(repeat_num, 11);
inv_MinLoss = nan(repeat_num, 26);

inv_MaxAccur = nan(repeat_num, 26);
noninv_MaxAccur = nan(repeat_num, 11);




for repeat = 1:repeat_num
    for lami = 1:length(lambds)
        disp(['repeat ' num2str(repeat) ' iteration ' num2str(lami)])
        
        [noninv_mdl] = fitclinear(noninv_datmat', logic_outcm','ObservationsIn','columns',...
            'Learner', 'svm', ...
            'Regularization', 'lasso', 'batchSize', 16, 'OptimizeLearnRate', true, 'KFold',10,...
            'Lambda', lambds(lami));
        
        [inv_mdl] = fitclinear(inv_datmat', logic_outcm','ObservationsIn','columns',...
            'Learner', 'svm', ...
            'Regularization', 'lasso', 'batchSize', 16, 'OptimizeLearnRate', true, ...
            'Kfold', 10, 'Lambda', lambds(lami));
        
        inv_cost(repeat, lami) = inv_mdl.kfoldLoss;
        noninv_cost(repeat, lami) = noninv_mdl.kfoldLoss;
        
        [inv_pred, inv_scores] = inv_mdl.kfoldPredict;
        [~,~,thres1, jjj1] = perfcurve(logic_outcm, inv_scores(:,1),0);
        [noninv_pred, noninv_scores] = noninv_mdl.kfoldPredict;
%         [~,~,~, noninv_AUC(repeat, lami)] 
        [X,Y,Thres,jjj] = perfcurve(logic_outcm, noninv_scores(:, 1), 0);
        
        TPinv = sum((logic_outcm == 0) & (inv_pred == 0));
        TNinv = sum((logic_outcm == 1) & (inv_pred == 1));
        TPnoninv = sum((logic_outcm == 0) & (noninv_pred == 0));
        TNnoninv = sum((logic_outcm == 1) & (noninv_pred == 1));
        inv_accuracy(repeat, lami) = (TPinv + TNinv)/(length(logic_outcm));
        noninv_accuracy(repeat, lami) = (TPnoninv + TNnoninv)/(length(logic_outcm));
        
        
        
        
        noninvBet = nan(10,11);
        for ii = 1:10
            temp_Bet = noninv_mdl.Trained{ii}.Beta;
            noninvBet(ii,:) = temp_Bet;
        end
        noninvBet = mean(noninvBet,1);
        noninv_non0num = length(find(abs(noninvBet)));
        
        
        invBet = nan(10,26);
        for ii = 1:10
            temp_Bet = inv_mdl.Trained{ii}.Beta;
            invBet(ii,:) = temp_Bet;
        end
        
        invBet = mean(invBet,1);
        inv_non0num = length(find(abs(invBet)));
        
        
        inv_size(repeat, lami) = inv_non0num;
        noninv_size(repeat, lami) = noninv_non0num;
        

    end % end of loop on lambda
    
    
    
%     for invi = 1:27
%         try
%             temp_ind = find(inv_size(repeat, :) == invi-1);
%             inv_MinLoss(repeat, invi) = min(inv_cost(repeat, temp_ind));
%         catch
%             continue
%         end
%     end
% %     
%     
%     
%     for invi = 1:12
%         try 
%             temp_ind = find(noninv_size(repeat, :) == uniq_noninv_sizes(invi));
%             noninv_MinLoss(repeat, invi) = min(noninv_cost(repeat, temp_ind));
%         catch
%             continue
%         end
%     end
%     
%     %-------------Choose maximum------
%     for invi = 1:27
%         try
%             temp_ind = find(inv_size(repeat, :) == uniq_inv_sizes(invi));
%             inv_MaxAccur(repeat, invi) = max(inv_accuracy(repeat, temp_ind));
%         catch 
%             continue
%         end
%     end
%     
% %     
%     
%     for invi = 1:12
%         try
%             temp_ind = find(noninv_size(repeat, :) == uniq_noninv_sizes(invi));
%             noninv_MaxAccur(repeat, invi) = max(noninv_accuracy(repeat, temp_ind));
%         catch
%             continue
%         end
%     end
%     
    end

%%


%--------------Defining Boundries -------

%----Lower bound----
noninv_LowerLoss = min(noninv_MinLoss,[], 1);
inv_LowerLoss = min(inv_MinLoss, [], 1);

noninv_LowerAccur = min(noninv_MaxAccur, [], 1);
inv_LowerAccur = min(inv_MaxAccur, [], 1);

%------Uper bound-----
noninv_UpperLoss = max(noninv_MinLoss,[], 1);
inv_UpperLoss = max(inv_MinLoss, [], 1);

noninv_UpperAccur = max(noninv_MaxAccur, [], 1);
inv_UpperAccur = max(inv_MaxAccur, [], 1);

%----------Std of cost and accuracy-----
noninv_StdLoss = std(noninv_MinLoss,[], 1);
inv_StdLoss = std(inv_MinLoss, [], 1);

noninv_StdAccur = std(noninv_MaxAccur, [], 1);
inv_StdAccur = std(inv_MaxAccur, [], 1);
%-----Mean Accur and coast-----
noninv_MeanLoss = mean(noninv_MinLoss, 1);
inv_MeanLoss = mean(inv_MinLoss, 1);

noninv_MeanAccur = mean(noninv_MaxAccur, 1);
inv_MeanAccur = mean(inv_MaxAccur, 1);
%-------------------------
%% 
inv_size_mod = mode(inv_size);
noninv_size_mod = mode(noninv_size);
inv_select_ind = [];
noninv_select_ind = [];

for iti = 1:length(inv_size_mod)
    if iti == 1
        inv_select_ind = [inv_select_ind iti];
        continue
    end
    if inv_size_mod(iti) == inv_size_mod(iti-1);
        continue
    else
        inv_select_ind = [inv_select_ind iti];
    end 
end

for iti = 1:length(noninv_size_mod)
    if iti == 1
        noninv_select_ind = [noninv_select_ind iti];
        continue
    end
    if noninv_size_mod(iti) == noninv_size_mod(iti-1);
        continue
    else
        noninv_select_ind = [noninv_select_ind iti];
    end 
end

%-----------Select costs and accuracies corresponding to indexes--------

plot_inv_mod = inv_size_mod(inv_select_ind);
plot_noninv_mod = noninv_size_mod(noninv_select_ind);

plot_inv_accuracy = inv_accuracy(:, inv_select_ind);
plot_noninv_accuracy = noninv_accuracy(:, noninv_select_ind);
%------
inv_accur_fin = [];
inv_accur_finStd = [];
inv_cost_fin = [];
inv_cost_finStd = [];
inv_AUC_fin = [];
inv_AUC_finStd = [];
%-----
for iteri = 1:length(inv_select_ind)
    temp_accur = inv_accuracy(find(inv_size(:, inv_select_ind(iteri))== plot_inv_mod(iteri)), inv_select_ind(iteri));
    inv_accur_fin = [inv_accur_fin mean(temp_accur)];
    inv_accur_finStd = [inv_accur_finStd std(temp_accur)];    
    temp_cost = inv_cost(find(inv_size(:, inv_select_ind(iteri))== plot_inv_mod(iteri)), inv_select_ind(iteri));
    inv_cost_fin = [inv_cost_fin mean(temp_cost)];
    inv_cost_finStd = [inv_cost_finStd std(temp_cost)];
    
    temp_AUC = inv_AUC(find(inv_size(:, inv_select_ind(iteri))== plot_inv_mod(iteri)), inv_select_ind(iteri));
    inv_AUC_fin = [inv_AUC_fin mean(temp_AUC)];
    inv_AUC_finStd = [inv_AUC_finStd std(temp_AUC)];
end

%------
noninv_accur_fin = [];
noninv_accur_finStd = [];
noninv_cost_fin = [];
noninv_cost_finStd = [];
noninv_AUC_fin = [];
noninv_AUC_finStd = [];
%------
for iteri = 1:length(noninv_select_ind)
    temp_accur = noninv_accuracy(find(noninv_size(:, noninv_select_ind(iteri))== plot_noninv_mod(iteri)), ...
        noninv_select_ind(iteri));
    noninv_accur_fin = [noninv_accur_fin mean(temp_accur)];
    noninv_accur_finStd = [noninv_accur_finStd std(temp_accur)];
    
    temp_cost = noninv_cost(find(noninv_size(:, noninv_select_ind(iteri))== plot_noninv_mod(iteri)), ...
        noninv_select_ind(iteri));
    noninv_cost_fin = [noninv_cost_fin mean(temp_cost)];
    noninv_cost_finStd = [noninv_cost_finStd std(temp_cost)];

    temp_AUC = noninv_AUC(find(noninv_size(:, noninv_select_ind(iteri))== plot_noninv_mod(iteri)), ...
        noninv_select_ind(iteri));
    noninv_AUC_fin = [noninv_AUC_fin mean(temp_AUC)];
    noninv_AUC_finStd = [noninv_AUC_finStd std(temp_AUC)];
end

%-------
% plot using noninv_accur_fin, noninv_accur_finStd, plot_noninv_mod,
% plot_inv_mod
% Reverse them 
noninv_accur_fin = noninv_accur_fin(end:-1:1);
noninv_accur_finStd = noninv_accur_finStd(end:-1:1);
inv_accur_fin = inv_accur_fin(end:-1:1);
inv_accur_finStd = inv_accur_finStd(end:-1:1);

plot_noninv_mod = plot_noninv_mod(end:-1:1);
plot_inv_mod = plot_inv_mod(end:-1:1);
%-------------------
figure
hold on
patch([plot_noninv_mod plot_noninv_mod(end:-1:1)],...
    [noninv_accur_fin-noninv_accur_finStd noninv_accur_fin(end:-1:1)+noninv_accur_finStd(end:-1:1)],...
    [0 0.5 0], 'FaceAlpha', 0.2, 'EdgeColor', [0 0.5 0])
patch([plot_noninv_mod plot_noninv_mod(end:-1:1)],...
    [noninv_accur_fin-noninv_accur_finStd noninv_accur_fin(end:-1:1)+noninv_accur_finStd(end:-1:1)],...
    [0 0.5 0], 'FaceAlpha', 0.2, 'EdgeColor', [0 0.5 0])
plot(plot_noninv_mod, noninv_accur_fin, '--o', 'Markeredgecolor', 'k',...
    'markerfacecolor', [0 0.5 0], 'markersize', 6, 'color', 'k');
 
patch([plot_inv_mod plot_inv_mod(end:-1:1)],...
    [inv_accur_fin-inv_accur_finStd inv_accur_fin(end:-1:1)+inv_accur_finStd(end:-1:1)],...
    'w','FaceColor', [0.7 0.7 0.7], 'FaceAlpha', 0.9,'EdgeColor', [0.5 0.5 0.5])
patch([plot_inv_mod plot_inv_mod(end:-1:1)],...
    [inv_accur_fin-inv_accur_finStd inv_accur_fin(end:-1:1)+inv_accur_finStd(end:-1:1)],...
    'w','FaceColor', [0.7 0.7 0.7], 'FaceAlpha', 0.9,'EdgeColor', [0.5 0.5 0.5])
plot(plot_inv_mod, inv_accur_fin, '--o', 'Markeredgecolor', 'k',...
    'markerfacecolor', [0.4 0.4 0.4], 'markersize', 6, 'Color', 'k' );

ax = gca;
ax.Box = 'off'
xlabel('Number of features')
ylabel('Accuracy')
axis square
hold off
grid on
%--------------
noninv_cost_fin = noninv_cost_fin(end:-1:1);
noninv_cost_finStd = noninv_cost_finStd(end:-1:1);
inv_cost_fin = inv_cost_fin(end:-1:1);
inv_cost_finStd = inv_cost_finStd(end:-1:1);

% plot_noninv_mod = plot_noninv_mod(end:-1:1);
% plot_inv_mod = plot_inv_mod(end:-1:1);
%-------------------
%---------Cost plot------
figure
hold on
patch([plot_noninv_mod plot_noninv_mod(end:-1:1)],...
    [noninv_cost_fin-noninv_cost_finStd noninv_cost_fin(end:-1:1)+noninv_cost_finStd(end:-1:1)],...
    [0 0.5 0], 'FaceAlpha', 0.2, 'EdgeColor', [0 0.5 0])
patch([plot_noninv_mod plot_noninv_mod(end:-1:1)],...
    [noninv_cost_fin-noninv_cost_finStd noninv_cost_fin(end:-1:1)+noninv_cost_finStd(end:-1:1)],...
    [0 0.5 0], 'FaceAlpha', 0.2, 'EdgeColor', [0 0.5 0])
plot(plot_noninv_mod, noninv_cost_fin, '--o', 'Markeredgecolor', 'k',...
    'markerfacecolor', [0 0.5 0], 'markersize', 6, 'color', 'k');
 
patch([plot_inv_mod plot_inv_mod(end:-1:1)],...
    [inv_cost_fin-inv_cost_finStd inv_cost_fin(end:-1:1)+inv_cost_finStd(end:-1:1)],...
    'w','FaceColor', [0.7 0.7 0.7], 'FaceAlpha', 0.9,'EdgeColor', [0.5 0.5 0.5])
patch([plot_inv_mod plot_inv_mod(end:-1:1)],...
    [inv_cost_fin-inv_cost_finStd inv_cost_fin(end:-1:1)+inv_cost_finStd(end:-1:1)],...
    'w','FaceColor', [0.7 0.7 0.7], 'FaceAlpha', 0.9,'EdgeColor', [0.5 0.5 0.5])
plot(plot_inv_mod, inv_cost_fin, '--o', 'Markeredgecolor', 'k',...
    'markerfacecolor', [0.4 0.4 0.4], 'markersize', 6, 'Color', 'k' );

ax = gca;
ax.Box = 'off';
xlabel('Number of features')
ylabel('Misclassification cost')
axis square
hold off
grid on
% save2pdf('sparsity-numfetures')
%----------AUC
noninv_AUC_fin = noninv_AUC_fin(end:-1:1);
noninv_AUC_finStd = noninv_AUC_finStd(end:-1:1);
inv_AUC_fin = inv_AUC_fin(end:-1:1);
inv_AUC_finStd = inv_AUC_finStd(end:-1:1);

%----AUC plot----
figure
hold on
patch([plot_noninv_mod plot_noninv_mod(end:-1:1)],...
    [noninv_AUC_fin-noninv_AUC_finStd noninv_AUC_fin(end:-1:1)+noninv_AUC_finStd(end:-1:1)],...
    [0 0.5 0], 'FaceAlpha', 0.2, 'EdgeColor', [0 0.5 0])
patch([plot_noninv_mod plot_noninv_mod(end:-1:1)],...
    [noninv_AUC_fin-noninv_AUC_finStd noninv_AUC_fin(end:-1:1)+noninv_AUC_finStd(end:-1:1)],...
    [0 0.5 0], 'FaceAlpha', 0.2, 'EdgeColor', [0 0.5 0])
plot(plot_noninv_mod, noninv_AUC_fin, '--o', 'Markeredgecolor', 'k',...
    'markerfacecolor', [0 0.5 0], 'markersize', 6, 'color', 'k');
 
patch([plot_inv_mod plot_inv_mod(end:-1:1)],...
    [inv_AUC_fin-inv_AUC_finStd inv_AUC_fin(end:-1:1)+inv_AUC_finStd(end:-1:1)],...
    'w','FaceColor', [0.7 0.7 0.7], 'FaceAlpha', 0.9,'EdgeColor', [0.5 0.5 0.5])
patch([plot_inv_mod plot_inv_mod(end:-1:1)],...
    [inv_AUC_fin-inv_AUC_finStd inv_AUC_fin(end:-1:1)+inv_AUC_finStd(end:-1:1)],...
    'w','FaceColor', [0.7 0.7 0.7], 'FaceAlpha', 0.9,'EdgeColor', [0.5 0.5 0.5])
plot(plot_inv_mod, inv_AUC_fin, '--o', 'Markeredgecolor', 'k',...
    'markerfacecolor', [0.4 0.4 0.4], 'markersize', 6, 'Color', 'k' );

ax = gca;
ax.Box = 'off';
xlabel('Number of features')
ylabel('Misclassification cost')
axis square
hold off
grid on




%%

noninv_Cost_plot = mean(noninv_cost,1);
noninv_Cost_Std = std(noninv_cost,[],1);
noninv_Cost_SEM = std(noninv_cost,[],1)/sqrt(repeat_num);

inv_Cost_plot = mean(inv_cost, 1);
inv_Cost_Std = std(inv_cost,[],1);
inv_Cost_SEM = std(inv_cost,[],1)/sqrt(repeat_num);


noninv_accuracy_plot = mean(noninv_accuracy,1);
noninv_accuracy_Std = std(noninv_accuracy,[],1);
noninv_accuracy_SEM = std(noninv_accuracy,[],1)/sqrt(repeat_num);

inv_accuracy_plot = mean(inv_accuracy, 1);
inv_accuracy_Std = std(inv_accuracy,[],1);
inv_accuracy_SEM = std(inv_accuracy,[],1)/sqrt(repeat_num);
%-------------------
figure
hold on
plot(lambds, noninv_size_plot, '--s', 'markersize', 8, 'markeredgecolor', 'k', 'markerfacecolor', [0 0.6 0],...
   'color', 'k' )
plot(lambds, inv_size_plot, '--s', 'markersize', 8, 'markeredgecolor', 'k', 'markerfacecolor', [0.4 0.4 0.4],...
   'color', 'k' )
ylabel('Number of features')
ylim([0 27])

figure
ax = gca
ax.YAxisLocation = 'right'
hold on
plot(lambds, noninv_accuracy_plot, '--o', 'markersize', 7, 'markerfacecolor', [0 0.5 0],...
    'color', 'k')
plot(lambds, inv_accuracy_plot, '--o', 'markersize', 7, 'markerfacecolor', [0.4 0.4 0.4],...
    'color', 'k')

patch([lambds lambds(end:-1:1)], [noninv_accuracy_plot-noninv_accuracy_Std,...
    noninv_accuracy_plot(end:-1:1)+noninv_accuracy_Std(end:-1:1)],...
    'w', 'FaceColor'  , [0 0.3 0], 'FaceAlpha', 0.2, 'EdgeColor', [0 0.3 0])
patch([lambds lambds(end:-1:1)], [inv_accuracy_plot-inv_accuracy_Std...
    ,inv_accuracy_plot(end:-1:1)+inv_accuracy_Std(end:-1:1)],...
    'w','FaceColor', [0.3 0.3 0.3],  'FaceAlpha', 0.2, 'EdgeColor', [0.3 0.3 0.3])
ylabel('Accuracy')
xlabel('Lambda')
ax = gca;
% ax.YAxis(2).Color = [0 0 0];
legend({'Number of non-invasive features with non-zero weights', 'Number of invasive features with non-zero weights',...
    'Non-invasive accuracy', 'Invasive accuracy'}, 'location', 'northeast', 'Box', 'off')
grid on
% save2pdf('sparsity Analysis Std')

%-----------Cost plot---------------
figure
hold on
plot(lambds, noninv_size_plot, '--s', 'markersize', 8, 'markeredgecolor', 'k', 'markerfacecolor', [0 0.6 0],...
   'color', 'k' )
plot(lambds, inv_size_plot, '--s', 'markersize', 8, 'markeredgecolor', 'k', 'markerfacecolor', [0.4 0.4 0.4],...
   'color', 'k' )
ylabel('Number of features')
ylim([0 27])

yyaxis right
hold on
plot(lambds, noninv_cost_plot, '--o', 'markersize', 7, 'markerfacecolor', [0 0.5 0],...
    'color', 'k')
plot(lambds, inv_cost_plot, '--o', 'markersize', 7, 'markerfacecolor', [0.4 0.4 0.4],...
    'color', 'k')

patch([lambds lambds(end:-1:1)], [noninv_cost_plot-noninv_cost_Std,...
    noninv_cost_plot(end:-1:1)+noninv_cost_Std(end:-1:1)],...
    'w', 'FaceColor'  , [0 0.3 0], 'FaceAlpha', 0.2, 'EdgeColor', [0 0.3 0])
patch([lambds lambds(end:-1:1)], [inv_cost_plot-inv_cost_Std...
    ,inv_cost_plot(end:-1:1)+inv_cost_Std(end:-1:1)],...
    'w','FaceColor', [0.3 0.3 0.3],  'FaceAlpha', 0.2, 'EdgeColor', [0.3 0.3 0.3])
ylabel('Accuracy')
xlabel('Lambda')
ax = gca;
% ax.YAxis(2).Color = [0 0 0];
legend({'Number of non-invasive features with non-zero weights', 'Number of invasive features with non-zero weights',...
    'Non-invasive accuracy', 'Invasive accuracy'}, 'location', 'northeast', 'Box', 'off')
grid on
% save2pdf('sparsity Analysis Std')


%----------------
noninv_accuracy_CI = prctile(noninv_accuracy, [2.5 97.5]);
noninv_accuracy_median = median(noninv_accuracy);
inv_accuracy_CI = prctile(inv_accuracy, [2.5 97.5]);
inv_accuracy_median = median(inv_accuracy);

figure
hold on
plot(lambds, noninv_size_plot, '--s', 'markersize', 8, 'markeredgecolor', 'k', 'markerfacecolor', [0 0.6 0],...
   'color', 'k' )
plot(lambds, inv_size_plot, '--s', 'markersize', 8, 'markeredgecolor', 'k', 'markerfacecolor', [0.4 0.4 0.4],...
   'color', 'k' )
ylabel('Number of features')
ylim([0 27])
figure
% yyaxis right 
plot(lambds, noninv_accuracy_median , '--o', 'markersize', 7, 'markerfacecolor', [0 0.5 0],...
    'color', 'k')
plot(lambds, inv_accuracy_median , '--o', 'markersize', 7, 'markerfacecolor', [0.4 0.4 0.4],...
    'color', 'k')

patch([lambds lambds(end:-1:1)], [noninv_accuracy_CI(1,:) noninv_accuracy_CI(2,:)],...
    [0 0.5 0], 'FaceAlpha', 0.2, 'EdgeColor', [0 0.5 0])
patch([lambds lambds(end:-1:1)], [inv_accuracy_CI(1,:) inv_accuracy_CI(2,:)],...
    [0 0.5 0], 'FaceAlpha', 0.2, 'EdgeColor', [0.4 0.4 0.4])
ylabel('Accuracy')
xlabel('Lambda')
ax = gca;
% ax.YAxis(2).Color = [0 0 0];
legend({'Number of non-invasive features with non-zero weights', 'Number of invasive features with non-zero weights',...
    'Non-invasive accuracy', 'Invasive accuracy'}, 'location', 'northeast', 'Box', 'off')
grid on


%------------------
figure
plot(1:11, noninv_MeanLoss(2:end), '--o', 'Markeredgecolor', 'b',...
    'markerfacecolor', 'g', 'markersize', 6);
hold on
plot(1:26, inv_MeanLoss(2:end), '--o', 'Markeredgecolor', 'b',...
    'markerfacecolor', [0.3 0.3 0.3], 'markersize', 6);
hold off



