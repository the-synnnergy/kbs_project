from ast import Index
import json
from random import seed
from matplotlib.font_manager import json_dump
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, KBinsDiscretizer
import sklearn.metrics
import sklearn.ensemble
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import corels
from sklearn import svm, tree
from sklearn.neural_network import MLPClassifier
import wittgenstein as lw
import imodels
import numpy as np
import gosdt
from imblearn.over_sampling import SMOTE
# Constants (seed for random state)
seed = 42

def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[feature_to_encode],prefix=feature_to_encode)
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res) 

def write_metrics_binary_classification_smote(clf, file_to_write, X, y):
    kf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    acc_scores = []
    f1_scores = []
    roc_auc_scores = []
    recall_scores = []
    precision_scores = []
    confusion_matrices = []
    file = "./results/" + str(file_to_write)
    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        #print(X_train)
        sm = SMOTE(random_state= seed)
        X_train, y_train = sm.fit_resample(X_train,y_train)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc_scores.append(clf.score(X_test,y_test))
        f1_scores.append(f1_score(y_test,y_pred))
        roc_auc_scores.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
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
        f.write("ROC_AUC:"+json.dumps(roc_auc_scores)+str(sum(roc_auc_scores)/len(roc_auc_scores))+"\n")
        f.write("Recall:"+json.dumps(recall_scores)+str(sum(recall_scores)/len(recall_scores))+"\n")
        f.write("Precision"+json.dumps(precision_scores)+str(sum(precision_scores)/len(precision_scores))+"\n")
        f.write("Confusion Matrices:"+json.dumps(confusion_matrices)+"\n")

def write_metrics_binary_classification(clf, file_to_write, X, y):
    kf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    acc_scores = []
    f1_scores = []
    roc_auc_scores = []
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
        roc_auc_scores.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
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
        f.write("ROC_AUC:"+json.dumps(roc_auc_scores)+str(sum(roc_auc_scores)/len(roc_auc_scores))+"\n")
        f.write("Recall:"+json.dumps(recall_scores)+str(sum(recall_scores)/len(recall_scores))+"\n")
        f.write("Precision"+json.dumps(precision_scores)+str(sum(precision_scores)/len(precision_scores))+"\n")
        f.write("Confusion Matrices:"+json.dumps(confusion_matrices)+"\n")

def write_metrics_multi_classification(clf, file_to_write, X, y):
    kf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    acc_scores = []
    roc_auc_scores = []
    recall_scores = []
    precision_scores = []
    f1_scores = []
    file = "./results/" + str(file_to_write)
    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        sm = SMOTE(random_state= seed)
        X_train, y_train = sm.fit_resample(X_train,y_train)
        #print(X_train)
        clf.fit(X_train, y_train)
        acc_scores.append(clf.score(X_test,y_test))
        roc_auc_scores.append(roc_auc_score(y_test, clf.predict_proba(X_test),multi_class="ovr"))
        recall_scores.append(recall_score(y_test,clf.predict(X_test),average="macro"))
        precision_scores.append(precision_score(y_test,clf.predict(X_test),average="weighted"))
        f1_scores.append(f1_score(y_test,clf.predict(X_test),average="macro"))

        #print(clf.score(X_test,y_test))
    #print(X.head())
    #print(y.head())
    with open(file,"w") as f:
        f.write(json.dumps(acc_scores)+str(sum(acc_scores)/len(acc_scores))+"\n")
        f.write(json.dumps(f1_scores)+str(sum(f1_scores)/len(f1_scores))+"\n")
        f.write(json.dumps(roc_auc_scores)+str(sum(roc_auc_scores)/len(roc_auc_scores))+"\n")
        f.write(json.dumps(recall_scores)+str(sum(recall_scores)/len(recall_scores))+"\n")
        f.write(json.dumps(precision_scores)+str(sum(precision_scores)/len(precision_scores))+"\n")
if __name__ == "__main__": 
    # import datasets
    titanic_df = pd.read_csv("./datasets/titanic.csv")
    heart_df = pd.read_csv("./datasets/heart.csv")
    wine_red_df = pd.read_csv("./datasets/winequality-red.csv",sep=";")
    wine_white_df = pd.read_csv("./datasets/winequality-white.csv",sep=";")
    # clean dataframes of useless data ( titanic only) and get class as last element in df
    titanic_df = titanic_df.drop("Name",axis=1)
    titanic_df["Sex"] = titanic_df["Sex"].map({"female" : 1, "male" : 0}).astype(int)
    #print(titanic_df.head())
    ## Benchmarking
    rf = RandomForestClassifier(random_state=seed)
    #write_metrics_binary_classification(rf,"./results/rf_titanic.txt",titanic_df.drop(["Survived"],axis=1),titanic_df["Survived"]) 
    # binning
    titanic_df_le = titanic_df
    titanic_df_le["Fare"] = pd.qcut(titanic_df["Fare"],5,labels=False)
    titanic_df_le["Age"] = pd.qcut(titanic_df["Age"],10,labels=False)
    #print(titanic_df.head())
    le = LabelEncoder()
    titanic_df_le = titanic_df_le.apply(le.fit_transform)
    titanic_df_hot_one = titanic_df_le
    titanic_features = ["Fare","Age", "Siblings/Spouses Aboard","Parents/Children Aboard","Pclass"]
    for feature in titanic_features:
        titanic_df_hot_one = encode_and_bind(titanic_df_hot_one,feature)
    col_list = list(titanic_df)
    col_list[0], col_list[-1] = col_list[-1], col_list[0]
    titanic_df = titanic_df[col_list]
    #print(titanic_df.head())
    #
    col_list = list(titanic_df_hot_one)
    col_list[0], col_list[-1] = col_list[-1], col_list[0]
    titanic_df_hot_one = titanic_df_hot_one[col_list]
    #print(titanic_df_hot_one.head())
    # preprocess heart_csv
    heart_df_binned = heart_df
    heart_df_binned['age'] = pd.to_numeric(pd.cut(heart_df['age'],9,labels=[0,1,2,3,4,5,6,7,8]),downcast="float")
    heart_df_binned['trestbps'] = pd.to_numeric(pd.cut(heart_df['trestbps'],10,labels=[0,1,2,3,4,5,6,7,8,9]),downcast="float")
    heart_df_binned['chol'] = pd.to_numeric(pd.cut(heart_df['chol'],7,labels=[0,1,2,3,4,5,6]),downcast="float")
    heart_df_binned['thalach'] = pd.to_numeric((pd.cut(heart_df['thalach'],10,labels=[0,1,2,3,4,5,6,7,8,9])),downcast="float")
    heart_df_binned['oldpeak'] = pd.to_numeric((pd.cut(heart_df['oldpeak'],7,labels=[0,1,2,3,4,5,6])),downcast="float")
    le = LabelEncoder()
    heart_df_binned = heart_df_binned.apply(le.fit_transform)
    #print(heart_df_binned.head())
    heart_df_hot_one = heart_df_binned
    heart_features = ["age","sex","cp","trestbps","chol","thalach","oldpeak","slope","thal","ca","restecg"]
    for feature in heart_features:
        heart_df_hot_one = encode_and_bind(heart_df_hot_one,feature)
    #print(heart_df_hot_one.to_numpy())
    #encode_and_bind()
    #feature_list = list(heart_df_hot_one.drop(["target"]))
    X = heart_df_hot_one.drop(["target"],axis=1)
    #write_metrics_binary_classification(corels_clf, None, X,y)
    #write_metrics_binary_classification(corels_clf,None,titanic_df_hot_one.drop(["Survived"],axis=1),titanic_df_hot_one["Survived"])
    # preprocess wine white dataset
    #print(wine_white_df.head())
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
    print(wwine_quality.value_counts())
    print(rwine_quality.value_counts())
    #write_metrics_binary_classification(rf,"rftest.txt",heart_df.drop(["target"],axis=1),heart_df["target"])

    ## heart disease runs
    ##binary runs
    write_metrics_binary_classification(RandomForestClassifier(),"rf_heart_hot_one.txt",heart_df_hot_one.drop(["target"],axis=1),heart_df_hot_one["target"])
    write_metrics_binary_classification(svm.SVC(probability=True),"svm_heart_hot_one.txt",heart_df_hot_one.drop(["target"],axis=1),heart_df_hot_one["target"])
    write_metrics_binary_classification(MLPClassifier(max_iter=10000),"mlp_heart_hot_one.txt",heart_df_hot_one.drop(["target"],axis=1),heart_df_hot_one["target"])
    write_metrics_binary_classification(tree.DecisionTreeClassifier(),"dt_heart_hot_one.txt",heart_df_hot_one.drop(["target"],axis=1),heart_df_hot_one["target"])
    write_metrics_binary_classification(lw.RIPPER(),"ripper_heart_hot_one.txt",heart_df_hot_one.drop(["target"],axis=1),heart_df_hot_one["target"])
    write_metrics_binary_classification(imodels.OptimalRuleListClassifier(),"corels_heart_hot_one.txt",heart_df_hot_one.drop(["target"],axis=1),heart_df_hot_one["target"])
    try:
        write_metrics_binary_classification(imodels.OptimalTreeClassifier(time_limit=1200,regularization=0.035),"gosdt_heart_hot_one_035.txt",heart_df_hot_one.drop(["target"],axis=1),heart_df_hot_one["target"])
    except Exception as e:
        print("time out")
        print(e)
    ## titanic runs
    # titanic hot one
    write_metrics_binary_classification(RandomForestClassifier(),"rf_titanic_hot_one.txt",titanic_df_hot_one.drop(["Survived"],axis=1),titanic_df_hot_one["Survived"])
    write_metrics_binary_classification(svm.SVC(probability=True),"svm_titanic_hot_one.txt",titanic_df_hot_one.drop(["Survived"],axis=1),titanic_df_hot_one["Survived"])
    write_metrics_binary_classification(MLPClassifier(max_iter=10000),"mlp_titanic_hot_one.txt",titanic_df_hot_one.drop(["Survived"],axis=1),titanic_df_hot_one["Survived"])
    write_metrics_binary_classification(tree.DecisionTreeClassifier(),"dt_titanic_hot_one.txt",titanic_df_hot_one.drop(["Survived"],axis=1),titanic_df_hot_one["Survived"])
    write_metrics_binary_classification(lw.RIPPER(),"ripper_titanic_hot_one.txt",titanic_df_hot_one.drop(["Survived"],axis=1),titanic_df_hot_one["Survived"])
    write_metrics_binary_classification(imodels.OptimalRuleListClassifier(),"corels_titanic_hot_one.txt",titanic_df_hot_one.drop(["Survived"],axis=1),titanic_df_hot_one["Survived"])
    write_metrics_binary_classification(imodels.OptimalTreeClassifier(time_limit=1200),"gosdt_titanic_hot_one.txt",titanic_df_hot_one.drop(["Survived"],axis=1),titanic_df_hot_one["Survived"])
    ## white wine runs """

    # hot one with  medium, low and high
    write_metrics_binary_classification(RandomForestClassifier(),"rf_wwine_hot_one_low.txt",wine_white_df_hot_one,wwine_quality_hot_one["low"])
    write_metrics_binary_classification(svm.SVC(probability=True),"svm_wwine_hot_one_low.txt",wine_white_df_hot_one,wwine_quality_hot_one["low"])
    write_metrics_binary_classification(MLPClassifier(max_iter=10000),"mlp_wwine_hot_one_low.txt",wine_white_df_hot_one,wwine_quality_hot_one["low"])
    write_metrics_binary_classification(tree.DecisionTreeClassifier(),"dt_wwine_hot_one_low.txt",wine_white_df_hot_one,wwine_quality_hot_one["low"])
    write_metrics_binary_classification(lw.RIPPER(),"ripper_wwine_hot_one_low.txt",wine_white_df_hot_one,wwine_quality_hot_one["low"])
    write_metrics_binary_classification(imodels.OptimalRuleListClassifier(max_card=3),"corels_wwine_hot_one_low.txt",wine_white_df_hot_one,wwine_quality_hot_one["low"])
    #
    write_metrics_binary_classification(RandomForestClassifier(),"rf_wwine_hot_one_medium.txt",wine_white_df_hot_one,wwine_quality_hot_one["medium"])
    write_metrics_binary_classification(svm.SVC(probability=True),"svm_wwine_hot_one_medium.txt",wine_white_df_hot_one,wwine_quality_hot_one["medium"])
    write_metrics_binary_classification(MLPClassifier(max_iter=10000),"mlp_wwine_hot_one_medium.txt",wine_white_df_hot_one,wwine_quality_hot_one["medium"])
    write_metrics_binary_classification(tree.DecisionTreeClassifier(),"dt_wwine_hot_one_medium.txt",wine_white_df_hot_one,wwine_quality_hot_one["medium"])
    write_metrics_binary_classification(lw.RIPPER(),"ripper_wwine_hot_one_medium.txt",wine_white_df_hot_one,wwine_quality_hot_one["medium"])
    write_metrics_binary_classification(imodels.OptimalRuleListClassifier(max_card=3),"corels_wwine_hot_one_medium.txt",wine_white_df_hot_one,wwine_quality_hot_one["medium"])
    # 
    write_metrics_binary_classification(RandomForestClassifier(),"rf_wwine_hot_one_high.txt",wine_white_df_hot_one,wwine_quality_hot_one["high"])
    write_metrics_binary_classification(svm.SVC(probability=True),"svm_wwine_hot_one_high.txt",wine_white_df_hot_one,wwine_quality_hot_one["high"])
    write_metrics_binary_classification(MLPClassifier(max_iter=10000),"mlp_wwine_hot_one_high.txt",wine_white_df_hot_one,wwine_quality_hot_one["high"])
    write_metrics_binary_classification(tree.DecisionTreeClassifier(),"dt_wwine_hot_one_high.txt",wine_white_df_hot_one,wwine_quality_hot_one["high"])
    write_metrics_binary_classification(lw.RIPPER(),"ripper_wwine_hot_one_high.txt",wine_white_df_hot_one,wwine_quality_hot_one["high"])
    write_metrics_binary_classification(imodels.OptimalRuleListClassifier(max_card=3),"corels_wwine_hot_one_high.txt",wine_white_df_hot_one,wwine_quality_hot_one["high"])
    ## red wine runs
    write_metrics_multi_classification(RandomForestClassifier(),"rf_rwine_unbinned.txt",wine_red_df.drop(["quality"],axis=1),rwine_quality)
    write_metrics_multi_classification(svm.SVC(probability=True),"svm_rwine_unbinned.txt",wine_red_df.drop(["quality"],axis=1),rwine_quality)
    write_metrics_multi_classification(MLPClassifier(max_iter=10000),"mlp_rwine_unbinned.txt",wine_red_df.drop(["quality"],axis=1),rwine_quality)
    write_metrics_multi_classification(tree.DecisionTreeClassifier(),"dt_rwine_unbinned.txt",wine_red_df.drop(["quality"],axis=1),rwine_quality)
    write_metrics_binary_classification(RandomForestClassifier(),"rf_rwine_hot_one_low.txt",wine_red_df_hot_one,rwine_quality_hot_one["low"])
    write_metrics_binary_classification(svm.SVC(probability=True),"svm_rwine_hot_one_low.txt",wine_red_df_hot_one,rwine_quality_hot_one["low"])
    write_metrics_binary_classification(MLPClassifier(max_iter=10000),"mlp_rwine_hot_one_low.txt",wine_red_df_hot_one,rwine_quality_hot_one["low"])
    write_metrics_binary_classification(tree.DecisionTreeClassifier(),"dt_rwine_hot_one_low.txt",wine_red_df_hot_one,rwine_quality_hot_one["low"])
    write_metrics_binary_classification(lw.RIPPER(),"ripper_rwine_hot_one_low.txt",wine_red_df_hot_one,rwine_quality_hot_one["low"])
    write_metrics_binary_classification(imodels.OptimalRuleListClassifier(max_card=3),"corels_rwine_hot_one_low.txt",wine_red_df_hot_one,rwine_quality_hot_one["low"])
    #
    write_metrics_binary_classification(RandomForestClassifier(),"rf_rwine_hot_one_medium.txt",wine_red_df_hot_one,rwine_quality_hot_one["medium"])
    write_metrics_binary_classification(svm.SVC(probability=True),"svm_rwine_hot_one_medium.txt",wine_red_df_hot_one,rwine_quality_hot_one["medium"])
    write_metrics_binary_classification(MLPClassifier(max_iter=10000),"mlp_rwine_hot_one_medium.txt",wine_red_df_hot_one,rwine_quality_hot_one["medium"])
    write_metrics_binary_classification(tree.DecisionTreeClassifier(),"dt_rwine_hot_one_medium.txt",wine_red_df_hot_one,rwine_quality_hot_one["medium"])
    write_metrics_binary_classification(lw.RIPPER(),"ripper_rwine_hot_one_medium.txt",wine_red_df_hot_one,rwine_quality_hot_one["medium"])
    write_metrics_binary_classification(imodels.OptimalRuleListClassifier(max_card=3),"corels_rwine_hot_one_medium.txt",wine_red_df_hot_one,rwine_quality_hot_one["medium"])
    # 
    write_metrics_binary_classification(RandomForestClassifier(),"rf_rwine_hot_one_high.txt",wine_red_df_hot_one,rwine_quality_hot_one["high"])
    write_metrics_binary_classification(svm.SVC(probability=True),"svm_rwine_hot_one_high.txt",wine_red_df_hot_one,rwine_quality_hot_one["high"])
    write_metrics_binary_classification(MLPClassifier(max_iter=10000),"mlp_rwine_hot_one_high.txt",wine_red_df_hot_one,rwine_quality_hot_one["high"])
    write_metrics_binary_classification(tree.DecisionTreeClassifier(),"dt_rwine_hot_one_high.txt",wine_red_df_hot_one,rwine_quality_hot_one["high"])
    write_metrics_binary_classification(lw.RIPPER(),"ripper_rwine_hot_one_high.txt",wine_red_df_hot_one,rwine_quality_hot_one["high"])
    write_metrics_binary_classification(imodels.OptimalRuleListClassifier(max_card=3),"corels_rwine_hot_one_high.txt",wine_red_df_hot_one,rwine_quality_hot_one["high"])

    # WWINE SMOTE
    write_metrics_binary_classification_smote(RandomForestClassifier(),"rf_wwine_hot_one_high_smote.txt",wine_white_df_binned,wwine_quality_hot_one["high"])
    write_metrics_binary_classification_smote(svm.SVC(probability=True),"svm_wwine_hot_one_high_smote.txt",wine_white_df_binned,wwine_quality_hot_one["high"])
    write_metrics_binary_classification_smote(MLPClassifier(max_iter=10000),"mlp_wwine_hot_one_high_smote.txt",wine_white_df_binned,wwine_quality_hot_one["high"])
    write_metrics_binary_classification_smote(tree.DecisionTreeClassifier(),"dt_wwine_hot_one_high_smote.txt",wine_white_df_binned,wwine_quality_hot_one["high"])
    write_metrics_binary_classification_smote(lw.RIPPER(n_discretize_bins=5),"ripper_wwine_hot_one_high_smote.txt",wine_white_df_binned,wwine_quality_hot_one["high"])
    write_metrics_binary_classification_smote(imodels.OptimalRuleListClassifier(max_card=3),"corels_wwine_hot_one_high_smote.txt",wine_white_df_hot_one,wwine_quality_hot_one["high"])
    # red wine smote
    # low
    write_metrics_binary_classification_smote(RandomForestClassifier(),"rf_rwine_hot_one_low_smote.txt",wine_red_df_binned,rwine_quality_hot_one["low"])
    write_metrics_binary_classification_smote(svm.SVC(probability=True),"svm_rwine_hot_one_low_smote.txt",wine_red_df_binned,rwine_quality_hot_one["low"])
    write_metrics_binary_classification_smote(MLPClassifier(max_iter=10000),"mlp_rwine_hot_one_low_smote.txt",wine_red_df_binned,rwine_quality_hot_one["low"])
    write_metrics_binary_classification_smote(tree.DecisionTreeClassifier(),"dt_rwine_hot_one_low_smote.txt",wine_red_df_binned,rwine_quality_hot_one["low"])
    write_metrics_binary_classification_smote(lw.RIPPER(n_discretize_bins=10),"ripper_rwine_hot_one_low_smote.txt",wine_red_df_hot_one,rwine_quality_hot_one["low"])
    write_metrics_binary_classification_smote(imodels.OptimalRuleListClassifier(max_card=3),"corels_rwine_hot_one_low_smote.txt",wine_red_df_hot_one,rwine_quality_hot_one["low"])
    # high
    write_metrics_binary_classification_smote(RandomForestClassifier(),"rf_rwine_hot_one_high_smote.txt",wine_red_df_binned,rwine_quality_hot_one["high"])
    write_metrics_binary_classification_smote(svm.SVC(probability=True),"svm_rwine_hot_one_high_smote.txt",wine_red_df_binned,rwine_quality_hot_one["high"])
    write_metrics_binary_classification_smote(MLPClassifier(max_iter=10000),"mlp_rwine_hot_one_high_smote.txt",wine_red_df_binned,rwine_quality_hot_one["high"])
    write_metrics_binary_classification_smote(tree.DecisionTreeClassifier(),"dt_rwine_hot_one_high_smote.txt",wine_red_df_binned,rwine_quality_hot_one["high"])
    write_metrics_binary_classification_smote(lw.RIPPER(n_discretize_bins=5),"ripper_rwine_hot_one_high_smote.txt",wine_red_df_binned,rwine_quality_hot_one["high"])
    write_metrics_binary_classification_smote(imodels.OptimalRuleListClassifier(max_card=3),"corels_rwine_hot_one_high_smote.txt",wine_red_df_hot_one,rwine_quality_hot_one["high"])