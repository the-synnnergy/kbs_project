import pandas as pd
import numpy as np
from model.gosdt import GOSDT
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, KBinsDiscretizer
import sklearn.metrics
import sklearn.ensemble
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import json
from imblearn.over_sampling import SMOTE

def write_metrics_binary_classification_smote(clf, file_to_write, X, y):
    kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    acc_scores = []
    f1_scores = []
    #roc_auc_scores = []
    recall_scores = []
    precision_scores = []
    confusion_matrices = []
    file = "./results/" + str(file_to_write)
    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        #print(X_train)
        sm = SMOTE(random_state= 42)
        X_train, y_train = sm.fit_resample(X_train,y_train)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc_scores.append(clf.score(X_test,y_test))
        f1_scores.append(f1_score(y_test,y_pred))
        #roc_auc_scores.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
        recall_scores.append(recall_score(y_test,y_pred))
        precision_scores.append(precision_score(y_test,y_pred))
        confusion_matrices.append(np.array2string(confusion_matrix(y_test,y_pred)))
        #print(clf.score(X_test,y_test))
    #print(X.head())
    #print(y.head())
    #clf.fit(X,y)
    print(clf.score(X,y))
    with open(file,"w") as f:
        f.write("Acc:"+json.dumps(acc_scores)+str(sum(acc_scores)/len(acc_scores))+"\n")
        f.write("F1:"+json.dumps(f1_scores)+str(sum(f1_scores)/len(f1_scores))+"\n")
        #f.write("ROC_AUC:"+json.dumps(roc_auc_scores)+str(sum(roc_auc_scores)/len(roc_auc_scores))+"\n")
        f.write("Recall:"+json.dumps(recall_scores)+str(sum(recall_scores)/len(recall_scores))+"\n")
        f.write("Precision"+json.dumps(precision_scores)+str(sum(precision_scores)/len(precision_scores))+"\n")
        f.write("Confusion Matrices:"+json.dumps(confusion_matrices)+"\n")

def write_metrics_multi_classification(clf, file_to_write, X, y):
    kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    acc_scores = []
    recall_scores = []
    precision_scores = []
    f1_scores = []
    file = "./results/" + str(file_to_write)
    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        #print(X_train)
        sm = SMOTE(random_state= 42)
        X_train, y_train = sm.fit_resample(X_train,y_train)
        clf.fit(X_train, y_train)
        acc_scores.append(clf.score(X_test,y_test))
        recall_scores.append(recall_score(y_test,clf.predict(X_test),average="macro"))
        precision_scores.append(precision_score(y_test,clf.predict(X_test),average="weighted"))
        f1_scores.append(f1_score(y_test,clf.predict(X_test),average="macro"))
        print(clf)
        #print(clf.score(X_test,y_test))
    #print(X.head())
    #print(y.head())
    with open(file,"w") as f:
        f.write(json.dumps(acc_scores)+str(sum(acc_scores)/len(acc_scores))+"\n")
        f.write(json.dumps(f1_scores)+str(sum(f1_scores)/len(f1_scores))+"\n")
        f.write(json.dumps(recall_scores)+str(sum(recall_scores)/len(recall_scores))+"\n")
        f.write(json.dumps(precision_scores)+str(sum(precision_scores)/len(precision_scores))+"\n")
def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[feature_to_encode],prefix=feature_to_encode)
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res) 

def write_metrics_binary_classification(clf, file_to_write, X, y):
    kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    acc_scores = []
    f1_scores = []
    #roc_auc_scores = []
    recall_scores = []
    precision_scores = []
    confusion_matrices = []
    file = "./results/" + str(file_to_write)
    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        #print(X_train)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc_scores.append(clf.score(X_test,y_test))
        f1_scores.append(f1_score(y_test,y_pred))
        #roc_auc_scores.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
        recall_scores.append(recall_score(y_test,y_pred))
        precision_scores.append(precision_score(y_test,y_pred))
        confusion_matrices.append(np.array2string(confusion_matrix(y_test,y_pred)))
        print(clf.tree)
    #print(X.head())
    #print(y.head())
    #clf.fit(X,y)
    print(clf.score(X,y))
    with open(file,"w") as f:
        f.write("Acc:"+json.dumps(acc_scores)+str(sum(acc_scores)/len(acc_scores))+"\n")
        f.write("F1:"+json.dumps(f1_scores)+str(sum(f1_scores)/len(f1_scores))+"\n")
        #f.write("ROC_AUC:"+json.dumps(roc_auc_scores)+str(sum(roc_auc_scores)/len(roc_auc_scores))+"\n")
        f.write("Recall:"+json.dumps(recall_scores)+str(sum(recall_scores)/len(recall_scores))+"\n")
        f.write("Precision"+json.dumps(precision_scores)+str(sum(precision_scores)/len(precision_scores))+"\n")
        f.write("Confusion Matrices:"+json.dumps(confusion_matrices)+"\n")

hyperparameters = {
    "regularization": 0.05,
    "time_limit": 1200,
    "verbose": True
}
wine_red_df = pd.read_csv("./winequality-red.csv",sep=";")
wine_white_df = pd.read_csv("./winequality-white.csv",sep=";")
titanic_df = pd.read_csv("./titanic.csv")
titanic_df = titanic_df.drop("Name",axis=1)
titanic_df["Sex"] = titanic_df["Sex"].map({"female" : 1, "male" : 0}).astype(int)
titanic_df_le = titanic_df
titanic_df_le["Fare"] = pd.qcut(titanic_df["Fare"],5,labels=False)
titanic_df_le["Age"] = pd.qcut(titanic_df["Age"],10,labels=False)
le = LabelEncoder()
titanic_df_le = titanic_df_le.apply(le.fit_transform)
titanic_df_hot_one = titanic_df_le
titanic_features = ["Fare","Age", "Siblings/Spouses Aboard","Parents/Children Aboard","Pclass"]
for feature in titanic_features:
    titanic_df_hot_one = encode_and_bind(titanic_df_hot_one,feature)
wwine_quality = wine_white_df["quality"]
wine_white_df = wine_white_df.drop(["quality"],axis=1)
#print(wwine_quality.value_counts())
wine_white_df_binned = wine_white_df
for feature in list(wine_white_df_binned):
    wine_white_df_binned[feature] = pd.qcut(wine_white_df_binned[feature],5,labels=False)
le = LabelEncoder()
wine_white_df_binned = wine_white_df_binned.apply(le.fit_transform)
#print(wine_white_df_binned.head())
wine_white_df_hot_one = wine_white_df_binned
for feature in list(wine_white_df_hot_one):
    wine_white_df_hot_one = encode_and_bind(wine_white_df_hot_one,feature)
wwine_quality = pd.cut(wwine_quality,3,labels=["low","medium","high"])
wwine_quality_hot_one = pd.get_dummies(wwine_quality)
# preprocess red wine dataset
rwine_quality = wine_red_df["quality"]
#print(rwine_quality.value_counts())
wine_red_df_binned = wine_red_df.drop(["quality"],axis=1)
for feature in list(wine_red_df_binned):
    wine_red_df_binned[feature] = pd.qcut(wine_red_df_binned[feature],5,labels=False)
le = LabelEncoder()
wine_red_df_binned = wine_red_df_binned.apply(le.fit_transform)
wine_red_df_hot_one = wine_red_df_binned
for feature in list(wine_red_df_binned):
    wine_red_df_hot_one = encode_and_bind(wine_red_df_hot_one,feature)
rwine_quality = pd.cut(rwine_quality,3,labels=["low","medium","high"])
rwine_quality_hot_one = pd.get_dummies(rwine_quality)
write_metrics_binary_classification(GOSDT(hyperparameters),"titanic_gosdt_reg.txt",titanic_df_hot_one.drop(["Survived"],axis=1),titanic_df_hot_one["Survived"])
write_metrics_multi_classification(GOSDT(hyperparameters),"white_wine_binned_gosdt.txt",wine_white_df,wwine_quality)
write_metrics_multi_classification(GOSDT(hyperparameters),"red_wine_binned_gosdt.txt",wine_red_df.drop(["quality"],axis=1),rwine_quality)
write_metrics_binary_classification_smote(GOSDT(hyperparameters),"white_wine_smote_high.txt",wine_white_df_hot_one,wwine_quality_hot_one["high"])
write_metrics_binary_classification_smote(GOSDT(hyperparameters),"red_wine_smote_high.txt",wine_red_df_hot_one,rwine_quality_hot_one["high"])
write_metrics_binary_classification_smote(GOSDT(hyperparameters),"red_wine_smote_low.txt",wine_red_df_hot_one,rwine_quality_hot_one["low"])