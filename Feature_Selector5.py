import os
import sys
import time
import math
import warnings
import multiprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import GenericUnivariateSelect, chi2, f_classif, mutual_info_classif, SelectFromModel, SequentialFeatureSelector, RFE, RFECV
from sklearn.feature_selection import SelectKBest, SelectFpr, SelectFdr, SelectFwe
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from MultiCollinearityEliminator import MultiCollinearityEliminator

def progressBar(count_value, total, prefix='Run', suffix='Ended'):
    bar_length = 100
    filled_up_Length = int(round(bar_length* count_value / float(total)))
    percentage = round(100.0 * count_value/float(total),1)
    bar = '>' * filled_up_Length + '-' * (bar_length - filled_up_Length)
    sys.stdout.write('%s [%s] %s%s %s\r' %(prefix,bar, percentage, '%', suffix))
    sys.stdout.flush()

def Calculate_Relevance(data, target, bins=10):
    Rel_main_df = pd.DataFrame()
    Rel_detail_df = pd.DataFrame()
    cols = data.columns
    My_y = data[target]
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'biufc') and (len(np.unique(data[ivars]))>bins):
            binned_x = pd.qcut(data[ivars], bins, duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': My_y})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': My_y})
        d = d0.groupby("x", as_index=False,observed=True).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['%'] = d['Events'] / d['N']
        d['Lift'] = d['%'] / My_y.mean()
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
        d['Importance'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        temp =pd.DataFrame({"Variable" : [ivars], "Importance" : [d['Importance'].sum()]}, columns = ["Variable", "Importance"])
        Rel_main_df=pd.concat([Rel_main_df,temp], axis=0)
        Rel_detail_df=pd.concat([Rel_detail_df,d], axis=0)
    Rel_main_df = Rel_main_df.set_index('Variable')
    Rel_main_df["Importance"]=np.where(Rel_main_df["Importance"]>1000,1000,Rel_main_df["Importance"])
    return Rel_main_df, Rel_detail_df

def MP(Model, X, y):
    My_scores = Model.predict_proba(X)
    My_predictions = Model.predict(X)
    TN, FP, FN, TP = confusion_matrix(y, My_predictions, labels=[0, 1]).ravel()
    accuracy    = (TP+TN)/(TP+TN+FP+FN)
    sensitivity = TP / (TP+FN)
    specifity   = TN / (TN+FP)
    if My_scores.shape[1]==2:
        fpr, tpr, thresholds = roc_curve(y, My_scores[:,1])
    else:
        fpr, tpr, thresholds = roc_curve(y, My_scores)
    AUC = auc(fpr, tpr)
    GINI = 2 * AUC - 1
    return AUC, GINI, accuracy, sensitivity, specifity

def Feature_Statistics(X_train,y_train):
    warnings.filterwarnings("ignore")
    nCPU = multiprocessing.cpu_count()
    os.environ["OMP_NUM_THREADS"] = str(nCPU-1)

    # collect statistics
    tmp=X_train.copy()
    tmp[y_train.name]=y_train
    IVs, WoEs = Calculate_Relevance(tmp,y_train.name)
    IVs.to_excel('IVs.xlsx')
    f_statistic, p_values = f_classif(X_train,y_train)
    chi2_stats, p_values = chi2(pd.DataFrame(MinMaxScaler().fit_transform(X_train),columns=X_train.columns),y_train)
    mi = mutual_info_classif(X_train,y_train,random_state=42)
    transformer = SelectFdr(score_func=f_classif)
    transformer.fit_transform(X_train, y_train)
    f_fdr = transformer.get_feature_names_out()
    transformer = SelectFpr(score_func=f_classif)
    transformer.fit_transform(X_train, y_train)    
    f_fpr = transformer.get_feature_names_out()
    transformer = SelectFwe(score_func=f_classif)
    transformer.fit_transform(X_train, y_train)
    f_fwe = transformer.get_feature_names_out()
    transformer = SelectFdr(score_func=chi2)
    transformer.fit_transform(pd.DataFrame(MinMaxScaler().fit_transform(X_train),columns=X_train.columns), y_train)
    chi2_fdr = transformer.get_feature_names_out()
    transformer = SelectFpr(score_func=chi2)
    transformer.fit_transform(pd.DataFrame(MinMaxScaler().fit_transform(X_train),columns=X_train.columns), y_train)    
    chi2_fpr = transformer.get_feature_names_out()
    transformer = SelectFwe(score_func=chi2)
    transformer.fit_transform(pd.DataFrame(MinMaxScaler().fit_transform(X_train),columns=X_train.columns), y_train)
    chi2_fwe = transformer.get_feature_names_out()
    selector = DecisionTreeClassifier(random_state=42,max_leaf_nodes=100,min_samples_leaf=50)
    selector.fit(X_train,y_train)
    FI_DT = selector.feature_importances_
    selector = LGBMClassifier(random_state=42,n_jobs=nCPU-1,max_leaf_nodes=100,min_samples_leaf=50,verbosity=-1)
    selector.fit(X_train,y_train)
    FI_LGBM = selector.feature_importances_
    selector = XGBClassifier(random_state=42,n_jobs=nCPU-1,max_leaf_nodes=100,min_samples_leaf=50)
    selector.fit(X_train,y_train)
    FI_XGB = selector.feature_importances_

    # put statistics
    df_features = pd.DataFrame(X_train.columns,columns=['feature'])
    df_features['IV'] = 0.0
    df_features['f'] = f_statistic
    df_features['chi2'] = chi2_stats
    df_features['mi'] = mi
    df_features['f_fdr'] = 0
    df_features['f_fpr'] = 0
    df_features['f_fwe'] = 0
    df_features['chi2_fdr'] = 0
    df_features['chi2_fpr'] = 0
    df_features['chi2_fwe'] = 0
    for idx, row in df_features.iterrows():
        df_features.loc[df_features['feature']==row['feature'],'IV'] = IVs.loc[IVs.index==row['feature'],'Importance'].iloc[0]
        if row['feature'] in f_fdr:
            df_features.loc[df_features['feature']==row['feature'],'f_fdr'] = 1
        else:
            df_features.loc[df_features['feature']==row['feature'],'f_fdr'] = 0
        if row['feature'] in f_fpr:
            df_features.loc[df_features['feature']==row['feature'],'f_fpr'] = 1
        else:
            df_features.loc[df_features['feature']==row['feature'],'f_fpr'] = 0
        if row['feature'] in f_fwe:
            df_features.loc[df_features['feature']==row['feature'],'f_fwe'] = 1
        else:
            df_features.loc[df_features['feature']==row['feature'],'f_fwe'] = 0        
        if row['feature'] in chi2_fdr:
            df_features.loc[df_features['feature']==row['feature'],'chi2_fdr'] = 1
        else:
            df_features.loc[df_features['feature']==row['feature'],'chi2_fdr'] = 0
        if row['feature'] in chi2_fpr:
            df_features.loc[df_features['feature']==row['feature'],'chi2_fpr'] = 1
        else:
            df_features.loc[df_features['feature']==row['feature'],'chi2_fpr'] = 0
        if row['feature'] in chi2_fwe:
            df_features.loc[df_features['feature']==row['feature'],'chi2_fwe'] = 1
        else:
            df_features.loc[df_features['feature']==row['feature'],'chi2_fwe'] = 0
    df_features['FI_DT'] = FI_DT
    df_features['FI_LGBM'] = FI_LGBM
    df_features['FI_XGB'] = FI_XGB

    df_features = df_features.fillna(0)
    df_features.to_excel('df_features.xlsx',index=False)

    df_features_n = df_features.copy()
    for col in df_features.columns:
        if col!='feature':
            if df_features_n[col].sum()==0:
                df_features_n[col] = 1 / df_features_n.shape[0]
            else:
                df_features_n[col] = df_features_n[col] / df_features_n[col].sum()
    df_features_n.to_excel('df_features_n.xlsx',index=False)
    
    return df_features_n

def Feature_Selector(X_train,y_train,k_best,df_features_n=None,param_set='Best'):
    all_params = pd.read_excel('Feature Selector Params.xlsx').set_index('param_set')
    params = all_params.loc[all_params.index==param_set,:].to_dict()
    rm_mult_thrs = params['rm_mult_thrs'][param_set]
    p1,p2,p3,p4  = params['p1'][param_set],params['p2'][param_set],params['p3'][param_set],params['p4'][param_set]
    p5,p6,p7,p8  = params['p5'][param_set],params['p6'][param_set],params['p7'][param_set],params['p8'][param_set]
    p9,p10       = params['p9'][param_set],params['p10'][param_set]
    p11,p12,p13  = params['p11'][param_set],params['p12'][param_set],params['p13'][param_set]
    selected = Feature_SelectorX(X_train,y_train,k_best,df_features_n,rm_mult_thrs,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13)
    return selected

def Feature_SelectorX(X_train,y_train,k_best,df_features_n,
                      rm_mult_thrs=0,p1=0,p2=0,p3=0,p4=0,p5=0,p6=0,p7=0,p8=0,p9=0,p10=0,p11=0,p12=0,p13=0):
    if df_features_n is None:
        df_features_n = Feature_Statistics(X_train,y_train)
    # STEP 0: Increase the quota
    n = max(1,int(k_best*1.5))
    # STEP 1: Select with respect to combined score
    df_features_n['score'] = df_features_n['IV'] * p1 + df_features_n['f'] * p2 + df_features_n['chi2'] * p3 +               \
                             df_features_n['mi'] * p4 + df_features_n['f_fdr'] * p5 + df_features_n['f_fpr'] * p6 +          \
                             df_features_n['f_fwe'] * p7 + df_features_n['chi2_fdr'] * p8 + df_features_n['chi2_fpr'] * p9 + \
                             df_features_n['chi2_fwe'] * p10 +                                                               \
                             df_features_n['FI_DT'] * p11 + df_features_n['FI_LGBM'] * p12 + df_features_n['FI_XGB'] * p13
    selected = df_features_n.nlargest(n,'score')['feature']
    # STEP 2: Remove Multicollinearity
    df = pd.merge(X_train[selected],y_train,left_index=True,right_index=True,how='inner')
    rm_multi_processor = MultiCollinearityEliminator(df,y_train.name,rm_mult_thrs,k_best)
    tmp = rm_multi_processor.autoEliminateMulticollinearity().drop(y_train.name,axis=1)
    selected = list(tmp.columns)
    # STEP 3: select the best wrt IV if more than quota or fill the quota if there is a gap
    if len(selected)>k_best:
        IVs = df_features_n[['feature','IV']].set_index('feature')
        selected = list(IVs[IVs.index.isin(selected)].nlargest(k_best,'IV').reset_index()['feature'])
    elif len(selected)<k_best:
        IVs = df_features_n[['feature','IV']].set_index('feature')
        c = 1
        while len(selected)<k_best:
            selected = list(set(selected)|set(list(IVs.nlargest(c,'IV').reset_index()['feature'])))
            c=c+1
    return selected


    