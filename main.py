from random import seed
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, KBinsDiscretizer
import sklearn.metrics
import sklearn.ensemble
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import corels

# Constants (seed for random state)
seed = 42

def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[feature_to_encode],prefix=feature_to_encode)
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res) 

def write_metrics_binary_classification(clf, file_to_write, X, y):
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    acc_scores = []
    f1_scores = []
    roc_auc_scores = []
    recall_scores = []
    precision_scores = []
    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        clf.fit(X_train, y_train)
        print(clf.score(X_test,y_test))
    print(X.head())
    print(y.head())
    clf.fit(X,y)
    print(clf.score(X,y))

if __name__ == "__main__": 
    # import datasets
    titanic_df = pd.read_csv("./datasets/titanic.csv")
    heart_df = pd.read_csv("./datasets/heart.csv")
    wine_red_df = pd.read_csv("./datasets/winequality-red.csv")
    wine_white_df = pd.read_csv("./datasets/winequality-white.csv")
    # clean dataframes of useless data ( titanic only) and get class as last element in df
    titanic_df = titanic_df.drop("Name",axis=1)
    titanic_df["Sex"] = titanic_df["Sex"].map({"female" : 1, "male" : 0}).astype(int)
    print(titanic_df.head())
    ## Benchmarking
    rf = RandomForestClassifier(random_state=seed)
    #write_metrics_binary_classification(rf,"./results/rf_titanic.txt",titanic_df.drop(["Survived"],axis=1),titanic_df["Survived"]) 
    # binning
    titanic_df["Fare"] = pd.qcut(titanic_df["Fare"],5,labels=False)
    titanic_df["Age"] = pd.qcut(titanic_df["Age"],10,labels=False)
    print(titanic_df.head())
    le = LabelEncoder()
    titanic_df_le = titanic_df.apply(le.fit_transform)
    titanic_df_hot_one = titanic_df_le
    features = ["Fare","Age", "Siblings/Spouses Aboard","Parents/Children Aboard"]
    for feature in features:
        titanic_df_hot_one = encode_and_bind(titanic_df_hot_one,feature)
    col_list = list(titanic_df)
    col_list[0], col_list[-1] = col_list[-1], col_list[0]
    titanic_df = titanic_df[col_list]
    print(titanic_df.head())
    #
    col_list = list(titanic_df_hot_one)
    col_list[0], col_list[-1] = col_list[-1], col_list[0]
    titanic_df_hot_one = titanic_df_hot_one[col_list]
    print(titanic_df_hot_one.head())
    #encode_and_bind()
    
