%% .............. Robustness of Mortality Risk Prediction....................
%....................... Mahdavi et.al 2020................
%..... A machine learning based exploration of COVID-19mortality risk...
% This script compute and compare the sensitivity of ROC curves to data setsample 
%
%
%   - Data: Represent the version of data. For example 
%           1: Old cohort, 2: Merged Cohort).
%   - ExtractData:  Load and trasfer Demographic and Labratory data 
%              into a matlab struct.
%              are typically the figure number.
%   - Save2pdf: Saves figure as a pdf with margins cropped to match 
%              the figure size.
%   - Bootstrap_size : Number of repeating itterations
%   - Comparing AUC : Final Result 
%   
%

%% Extract data
Data = 2; % 1:Old Cohort   2: Merged Cohort
[VarialbeNames,VariableMtrx,Category_Name,Category,data_strng] = ExtractData(Data);

model_color{1} = [0.5 0.5 0.5]; %Lab
model_color{2} = [0.1 0.4 0.1]; %Invasive
%%  SVM classifier for invasive and non-invasive data
h= figure(1);
scrsz = get(0,'ScreenSize');
scrsz(3) = scrsz(3)/2.6;
scrsz(4) = scrsz(4)/1;
set(h, 'Position',scrsz);

valid_features_Lab = find(Category==4);
valid_features_NonInvsv = find(Category<4);
full_features = find(Category<5);
Bootstrap_size = 30; % Number of itterations

Y = VariableMtrx(:,1);
X_full = VariableMtrx(:,1+[full_features]);
%..................... Prunning the incomplete data..................
[col, ~] = find(isnan(X_full));
vld_subj = setdiff([1:length(Y)],col);
Y = VariableMtrx(vld_subj,1);
X_full = VariableMtrx(vld_subj,1+[full_features]);
X = VariableMtrx(vld_subj,1+[valid_features_Lab]);
X_Wireless = VariableMtrx(vld_subj,1+[valid_features_NonInvsv]);
%....................... SVM model cross validation across itterations.............................................

for itter = 1:Bootstrap_size
    itter
    smpl = datasample([1:numel(Y)],floor(0.9*numel(Y)));
    
    SVMModel_full = fitcsvm(X_full(smpl,:),Y(smpl),'Standardize',true,...
        'KernelScale','auto','KernelFunction','linear');
    SVMModel = fitcsvm(X(smpl,:),Y(smpl),'Standardize',true,...
        'KernelScale','auto','KernelFunction','linear');
    SVMModel_Wireless = fitcsvm(X_Wireless(smpl,:),Y(smpl),'Standardize',true,...
        'KernelScale','auto','KernelFunction','linear');
    
    CVSVMModel_full = crossval(SVMModel_full,'kfold',5);
    CVSVMModel = crossval(SVMModel,'kfold',5);
    CVSVMModel_Wireless = crossval(SVMModel_Wireless,'kfold',5);
    
    classLoss_full = kfoldLoss(CVSVMModel_full);
    classLoss = kfoldLoss(CVSVMModel);
    classLoss_Wireless = kfoldLoss(CVSVMModel_Wireless);
    
    mdlSVM_full = fitPosterior(SVMModel_full);
    mdlSVM = fitPosterior(SVMModel);
    mdlSVM_Wireless = fitPosterior(SVMModel_Wireless);
    
    [~,score_svm_full] = resubPredict(mdlSVM_full);
    [~,score_svm] = resubPredict(mdlSVM);
    [~,score_svm_Wireless] = resubPredict(mdlSVM_Wireless);
    
    [Xsvm_full,Ysvm_full,Tsvm_full,AUCsvm_full,OPTROCPT_full] = perfcurve(Y(smpl),score_svm_full(:,1),0);
    [Xsvm,Ysvm,Tsvm,AUCsvm,OPTROCPT] = perfcurve(Y(smpl),score_svm(:,1),0);
    [Xsvm_Wireless,Ysvm_Wireless,Tsvm_Wireless,AUCsvm_Wireless,OPTROCPT_Wireless] = perfcurve(Y(smpl),score_svm_Wireless(:,1),0);
    
    TP_f(itter) = OPTROCPT_full(2);
    FP_f(itter) = OPTROCPT_full(1);
    
    TP(itter)= OPTROCPT(2);
    FP(itter) = OPTROCPT(1);
    
    TP_w(itter) = OPTROCPT_Wireless(2);
    FP_w(itter) = OPTROCPT_Wireless(1);
    
    Wire(itter) = AUCsvm_Wireless;
    Full(itter) =AUCsvm_full;
    Lab(itter) =AUCsvm;
end

%% Save and Print the Mean and SEM of TP,FP and AUC for each model

mean(Full)+i*2*std(Full)/sqrt(length(Full))
mean(Lab)+i*2*std(Lab)/sqrt(length(Lab))
mean(Wire)+i*2*std(Wire)/sqrt(length(Wire))

mean(TP_f)+i*2*std(TP_f)/sqrt(length(TP_f))
mean(TP)+i*2*std(TP)/sqrt(length(TP))
mean(TP_w)+i*2*std(TP_w)/sqrt(length(TP_w))

mean(FP_f)+i*2*std(FP_f)/sqrt(length(FP_f))
mean(FP)+i*2*std(FP)/sqrt(length(FP))
mean(FP_w)+i*2*std(FP_w)/sqrt(length(FP_w))
%% Plot Across Itteration Boxplot
h= figure(2);
scrsz = get(0,'ScreenSize');
scrsz(3) = scrsz(3)/2.6;
scrsz(4) = scrsz(4)/1;
set(h, 'Position',scrsz);

x_F = 1:length(Full);
x_W = 1:length(Wire);
x_L = 1:length(Lab);

CategoricalScatterplot([Full',Wire',Lab'],'Boxwidth',0.5)
hold on
legend(['Joint model (AUC = ',num2str(AUCsvm_full),')'],...
    ['Invasive model (AUC = ',num2str(AUCsvm),')'],...
    ['Wireless model (AUC = ',num2str(AUCsvm_Wireless),')'])

save2pdf(['Comparing AUC'])