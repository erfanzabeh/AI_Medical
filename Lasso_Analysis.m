%------ Generalized linear model with lasso regularization ------------
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

ran_ind = randperm(height(dat));
dat = dat(ran_ind,:);

dat_mat = dat(:, 2:end); %convert the input data to matrix format
%dat_cat = dat_mat(:, [2 9 10 11]);
%dat_mat(:, [2 9 10 11]) = [];
%dat_mat = [dat_mat dat_cat];

%---


var_nam = dat_mat.Properties.VariableNames;


dat_mat = table2array(dat_mat);
dat_mat = zscore(dat_mat);

logic_outcm = logical(table2array(dat(:, 1))); 

%% Model

[coef, fitinf] = lassoglm(dat_mat, logic_outcm , 'binomial', 'CV',10, 'PredictorNames',...
    var_nam); % Obtain the lasso generalized linear model


figure
lassoPlot(coef,fitinf,'plottype','CV'); 
legend('show', 'Box', 'off');
ax1 = gca;
ax1.Box = 'off';
xlim([0.001 0.1]);
%save2pdf('lassoplotCV')


idxLambda1SEDeviance = fitinf.Index1SE; % find the index of lambda where gives deviances within 1 SE of the min cost
mincoefs = find(coef(:,idxLambda1SEDeviance)); % find the index of non-zero coefs corresponding to the lambda
las_coefs = coef(:,idxLambda1SEDeviance); 
las_coefs = abs(las_coefs(las_coefs~=0)); % non-zero coefficient values of features corresponding to 1SE lambda
las_coefs = las_coefs/sum(las_coefs);

disp(var_nam(mincoefs)) %display the name of variables with non-zero coefs


predictors = mincoefs; % predictors of the model with non-zero coefficinets


 % to group non invasive and invasives 
 
%----- Plotting ------ 
figure
hold on
stem([1 2 3], las_coefs(1:3),'k--s', 'MarkerSize', 10, 'MarkerFaceColor', 'g')
stem([4 5 6 7], las_coefs(4:end),'k--s', 'MarkerSize', 10, 'MarkerFaceColor', 'g')
yli = get(gca, 'ylim');
patch([1 3.5 3.5 1], [yli(1) yli(1) 1 1], 'w','FaceColor', [0 0.5 0], 'FaceAlpha', 0.2, ...
    'EdgeColor', 'none')
patch([3.5 7 7 3.5], [yli(1) yli(1) 1 1], 'w','FaceColor', [0.4 0.4 0.4], 'FaceAlpha', 0.2, ...
    'EdgeColor', 'none')

hold off
temp_tick = var_nam(mincoefs);
xticklabels(temp_tick)
ylabel('Feature Coefficient')
xlim([-0.5 7.5])
save2pdf('LassoFeatures')

