# -*- coding: utf-8 -*-
"""Mortalite: feature setini genişletmenin dürüst AUC'ye etkisi (OOF-AUC + bootstrap %95 GA).
Eklenen değişkenler yalnızca t0'da bilindikleri klinik olarak doğrulandığında geçerlidir."""
import os, warnings, numpy as np, pandas as pd
from scipy.stats import randint, loguniform, uniform
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
try:
    from xgboost import XGBClassifier; HAS_XGB=True
except Exception: HAS_XGB=False
warnings.filterwarnings('ignore')
os.chdir(r"C:\Users\hasan\Documents\medical-research")
R=230
HONEST=['gebelikhaftası','cinsiyeti','apgar5','cerrahi','idrarsondasi','PNA_t0','crp1','proc1']

def load():
    df=pd.read_excel('micafunginson.xlsx', na_values=['#BOŞ!','#NULL!'])
    for c in ['kullandıgıtarih1','takibegirdigitarih','taburculuktarihi']:
        df[c]=pd.to_datetime(df[c],errors='coerce')
    df['PNA_t0']=(df['kullandıgıtarih1']-df['takibegirdigitarih']).dt.days
    df=df[(df['PNA_t0']>=0)&(df['PNA_t0']<=200)]
    df['time']=(df['taburculuktarihi']-df['kullandıgıtarih1']).dt.days
    df=df[df['time']>0].dropna(subset=['ex'])
    df['inotrop_severity']=df['ilkhaftainotrop'].map({0:0,1:1,2:1,3:2}).fillna(0)
    return df

def make_X(df,feats):
    X=df[feats].copy()
    for c in X.columns:
        X[c]=X[c].fillna(X[c].median())
    keep=[c for c in X.columns if X[c].nunique()>1]  # sabit sütunları at
    dropped=[c for c in X.columns if c not in keep]
    return X[keep].reset_index(drop=True), dropped

def models():
    m={"Logistic Regression":(LogisticRegression(random_state=R,max_iter=2000),{'C':loguniform(0.01,1000),'solver':['liblinear']}),
       "Random Forest":(RandomForestClassifier(random_state=R),{'n_estimators':randint(100,300),'max_depth':randint(3,7),'min_samples_split':randint(5,15),'max_features':['sqrt','log2',None]}),
       "SVC":(SVC(random_state=R,probability=True),{'C':loguniform(0.1,2000),'gamma':loguniform(1e-5,1e-1),'kernel':['rbf']}),
       "Gradient Boosting":(GradientBoostingClassifier(random_state=R),{'n_estimators':randint(50,300),'learning_rate':loguniform(0.005,0.1),'max_depth':randint(2,4),'subsample':uniform(0.6,0.4)}),
       "KNN":(KNeighborsClassifier(),{'n_neighbors':randint(3,11),'weights':['uniform','distance']}),
       "ExtraTrees":(ExtraTreesClassifier(random_state=R),{'n_estimators':randint(100,300),'max_depth':randint(3,7),'min_samples_split':randint(5,15),'max_features':['sqrt','log2',None]}),
       "AdaBoost":(AdaBoostClassifier(random_state=R),{'n_estimators':randint(50,200),'learning_rate':loguniform(0.01,0.5)}),
       "Decision Tree":(DecisionTreeClassifier(random_state=R),{'criterion':['gini','entropy'],'max_depth':randint(3,6),'min_samples_split':randint(5,20),'min_samples_leaf':randint(2,10)}),
       "MLP":(MLPClassifier(random_state=R,max_iter=2000,early_stopping=True),{'hidden_layer_sizes':[(4,),(8,),(12,)],'activation':['tanh','relu'],'solver':['lbfgs'],'alpha':loguniform(0.1,100)})}
    if HAS_XGB: m["XGBoost"]=(XGBClassifier(random_state=R,eval_metric='logloss'),{'n_estimators':randint(50,300),'learning_rate':loguniform(0.001,0.05),'max_depth':randint(2,4),'subsample':uniform(0.6,0.4),'colsample_bytree':uniform(0.6,0.4)})
    return m

def oof_auc(X,y,model,params):
    skf=StratifiedKFold(10,shuffle=True,random_state=R); yt,yp=[],[]
    for tri,tei in skf.split(X,y):
        Xtr,Xte=X.iloc[tri],X.iloc[tei]; ytr,yte=y.iloc[tri],y.iloc[tei]
        Xsm,ysm=SMOTE(random_state=42).fit_resample(Xtr,ytr)
        sc=StandardScaler(); Xtr_s,Xte_s=sc.fit_transform(Xsm),sc.transform(Xte)
        s=RandomizedSearchCV(model,params,n_iter=25,cv=3,n_jobs=-1,random_state=R,scoring='accuracy')
        s.fit(Xtr_s,ysm); yp.extend(s.best_estimator_.predict_proba(Xte_s)[:,1].tolist()); yt.extend(yte.tolist())
    return np.array(yt),np.array(yp)

def boot(yt,yp,B=2000):
    rng=np.random.default_rng(42); n=len(yt); a=[]
    for _ in range(B):
        i=rng.integers(0,n,n)
        if len(np.unique(yt[i]))>1: a.append(roc_auc_score(yt[i],yp[i]))
    return np.percentile(a,[2.5,97.5])

df=load(); y=df['ex'].astype(int).reset_index(drop=True)
SETS={
 "SET_8 (mevcut)":HONEST,
 "SET_8 + saf t0 (dogumagirligi, apgar1)":HONEST+['dogumagirligi','apgar1'],
 "SET_geniş (+entübe,inotrop,svk,tpn,dogumagirligi,apgar1)":HONEST+['entübe','inotrop_severity','svk','tpn','dogumagirligi','apgar1'],
}
print(f"Mortalite  n={len(y)}  ölüm={int(y.sum())}  sağ={int((y==0).sum())}\n")
for label,feats in SETS.items():
    X,dropped=make_X(df,feats)
    res=[]
    for name,(mdl,prm) in models().items():
        yt,yp=oof_auc(X,y,mdl,prm); res.append((name,roc_auc_score(yt,yp),yt,yp))
    res.sort(key=lambda r:r[1],reverse=True)
    champ,auc,yt,yp=res[0]; lo,hi=boot(yt,yp)
    drp=f"  (düşen sabit: {dropped})" if dropped else ""
    print(f"{label}  [{X.shape[1]} değişken]{drp}")
    print(f"   ŞAMPİYON: {champ:20s} OOF-AUC={auc:.3f}  %95 GA=[{lo:.3f}, {hi:.3f}]")
    print(f"   ilk 3: " + ", ".join(f'{n}={a:.3f}' for n,a,_,_ in res[:3]) + "\n", flush=True)
