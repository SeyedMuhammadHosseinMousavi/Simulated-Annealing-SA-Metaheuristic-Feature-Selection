
%% 
% The following code applies Simulated Annealing Evolutionary algorithm on a supervised
% Model as feature selection. Data is consisted of 300 samples from 6
% Classes (each class 50 samples) alongside with 40 features. The code reduces the features by half
% To 20 by selecting best features out of 40. Finally, KNN classification
% With proper confusion matrix plot, presents the performance of the
% System. You can load your data and define the desired number of features
% For it. Also, in order to better performance and depending on your system
% Power and your data, play with the SA parameters.
% If you find the code hard to understand, please feel free to contact me
%       Seyed Muhammad Hossein Mousavi
%       mosavi.a.i.buali@gmail.com

clc;
clear;

%% Start
% Loading
dat=load('fortest2.mat');
fordet=dat.FinalReady;
sizdet=size(fordet);
x=dat.FinalReady(:,1:sizdet(1,2)-1)';
t=dat.FinalReady(:,sizdet(1,2))';
nx=sizdet(1,2)-1;
nt=1;
nSample=sizdet(1,1);
% Converting Table to Struct
data.x=x;
data.t=t;
data.nx=nx;
data.nt=nt;
data.nSample=nSample;

% Number of Selected Features
nf=20;   
% Cost Function
CostFunction=@(q) FSC(q,nf,data);    

% Simulated Annealing Parameters
MaxIt=20;      % Max Number of Iterations
MaxSubIt=3;    % Max Number of Sub-iterations
T0=5;          % Initial Temp
alpha=0.99;    % Temp Reduction Rate

% Create and Evaluate Initial Solution
sol.Position=CRS(data);
[sol.Cost, sol.Out]=CostFunction(sol.Position);
% Initialize Best Solution Ever Found
BestSol=sol;
% Array to Hold Best Cost Values
BestCost=zeros(MaxIt,1);
% Intialize Temp.
T=T0;

%% Simulated Annealing Run
for it=1:MaxIt
for subit=1:MaxSubIt
% Create and Evaluate New Solution
        newsol.Position=NeighborCreation(sol.Position);
        [newsol.Cost, newsol.Out]=CostFunction(newsol.Position);        
        if newsol.Cost<=sol.Cost % If NEWSOL is better than SOL
        sol=newsol;            
        else % If NEWSOL is NOT better than SOL
        DELTA=(newsol.Cost-sol.Cost)/sol.Cost;            
        P=exp(-DELTA/T);
        if rand<=P
        sol=newsol;
        end    
        end 
% Update Best Solution Ever Found
    if sol.Cost<=BestSol.Cost
    BestSol=sol;
    end
    end
% Store Best Cost Ever Found
    BestCost(it)=BestSol.Cost;
% Display Iteration 
    disp(['In Iteration Number ' num2str(it) ': Best Cost Res = ' num2str(BestCost(it))]);
% Update Temp
    T=alpha*T;
end

% Plot Res
figure;
set(gcf, 'Position',  [450, 250, 900, 250])
plot(BestCost,'-.',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','r',...
    'Color',[0.9,0,0]);
title('Simulated Annealing')
xlabel('SA Iteration Number','FontSize',12,...
       'FontWeight','bold','Color','b');
ylabel('SA Best Cost Result','FontSize',12,...
       'FontWeight','bold','Color','b');
legend({'SA Train'});


%% Simulated Annealing Feature Selection (Final Step)
% Extracting Data
RealData=dat.FinalReady;
% Extracting Labels
RealLbl=RealData(:,end);
FinalFeaturesInd=BestSol.Out.S;
% Sort Features
FFI=sort(FinalFeaturesInd);
% Select Final Features
SA_Features=RealData(:,FFI);
% Adding Labels
SA_Features(:,end+1)=RealLbl;

%% Classification
% KNN Before SA
lblknn=RealData(:,end);
dataknn=RealData(:,1:end-1);
Mdl = fitcknn(dataknn,lblknn,'NumNeighbors',8,'Standardize',1);
rng(1); % For reproducibility
knndat = crossval(Mdl);
classError = kfoldLoss(knndat);
KNN = (1 - kfoldLoss(knndat, 'LossFun', 'ClassifError'))*100
% Predict the labels of the training data.
predictedknn = resubPredict(Mdl);

% KNN After SA
lblknn1=SA_Features(:,end);
dataknn1=SA_Features(:,1:end-1);
Mdl1 = fitcknn(dataknn1,lblknn1,'NumNeighbors',8,'Standardize',1);
rng(1); % For reproducibility
knndat1 = crossval(Mdl1);
classError1 = kfoldLoss(knndat1);
SA_KNN = (1 - kfoldLoss(knndat1, 'LossFun', 'ClassifError'))*100
% Predict the labels of the training data.
predictedknn1 = resubPredict(Mdl1);

% Confusion Matrix
figure
set(gcf, 'Position',  [150, 150, 1000, 350])
subplot(1,2,1)
cmknn = confusionchart(lblknn,predictedknn);
cmknn.Title = (['KNN on Original Data (Slower) =  ' num2str(KNN) '%']);
cmknn.RowSummary = 'row-normalized';
cmknn.ColumnSummary = 'column-normalized';
subplot(1,2,2)
cmknn1 = confusionchart(lblknn1,predictedknn1);
cmknn1.Title = (['KNN After SA Feature Selection (Faster) =  ' num2str(SA_KNN) '%']);
cmknn1.RowSummary = 'row-normalized';
cmknn1.ColumnSummary = 'column-normalized';
% ACC Res
fprintf('The (KNN on Original Data) Accuracy is = %0.4f.\n',KNN)
fprintf('The (KNN After SA Feature Selection) Accuracy is = %0.4f.\n',SA_KNN)

