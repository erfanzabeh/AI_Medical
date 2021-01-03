% ExtractData load cohort and define initial variables
%
%   [VarialbeNames,VariableMtrx,Category_Name,Category,data_strng]=
%   ExtractData(Data)
%
%   - Data: Represent the version of data. For example 
%           1: Old cohort, 2: Merged Cohort).
%
%   - VarialbeNames:  Abrevation of Labratory and Demographic data as a
%           1 by n cell that n represent the number of valid predictors.
%
%   - Category_Name: Type of Cohort Predictors. Demographic Features:1 
%          Clinical Features:2 Labratory:3
%

function  [VarialbeNames,VariableMtrx,Category_Name,Category,data_strng]= ExtractData(Data)

if Data==1 
M = readtable('data.xls');
data_strng = [];
else 
M = readtable('Additional_Data.xls');
data_strng = ['(Merged)'];
end

VarialbeNames = M.Properties.VariableNames;
VariableMtrx = M{:,:};

Abs_Neutr = VariableMtrx(:,13)*1000.*VariableMtrx(:,14)./100;
Abs_Lymph = VariableMtrx(:,13)*1000.*VariableMtrx(:,15)./100;

VariableMtrx(:,14) = Abs_Neutr;
VariableMtrx(:,15) = Abs_Lymph;

N = readtable('Features_Categorized.xlsx');

%.... Categorie databased on [Vital signs, Comorbidities, demographic, labratory]
Category = nan(size(VariableMtrx(:,2:end-1),2),1);
for feature = 1:37
    indx=find(ismember(N.Feature,VarialbeNames(1+feature)));
    Category(feature) = N.Class(indx);
end
Category(Category==2) = 1; % merge Vital Signs into Comorbidities...
Category_Name = {['Vital Signals'];['Comorbidities'];['Demographic'];['Labratory']};
end
