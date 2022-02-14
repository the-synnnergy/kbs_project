from random import seed
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sklearn.preprocessing
import sklearn.metrics
import sklearn.ensemble
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import corels

# Constants (seed for random state)
seed = 42

def write_metrics(clf, file_to_write, X, y):
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
    col_list = list(titanic_df)
    col_list[0], col_list[-1] = col_list[-1], col_list[0]
    titanic_df = titanic_df[col_list]
    titanic_df["Sex"] = titanic_df["Sex"].map({"female" : 1, "male" : 0}).astype(int)
    print(titanic_df.head())
    ## Benchmarking
    rf = RandomForestClassifier(random_state=seed)
    write_metrics(rf,"./results/rf_titanic.txt",titanic_df.drop(["Survived"],axis=1),titanic_df["Survived"])

