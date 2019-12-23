import pandas as pd
import category_encoders as ce
import matplotlib.pyplot as plt
import seaborn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def preprocessed(csv, show_details=False):
    df = pd.read_csv(csv)
    if show_details:
        print(df.head())
        print(df.columns)
        print(pd.isnull(df).any())
        unique_values = []
        for column in df.columns:
            unique_values.append([column, len(df[column].value_counts())])
        print(unique_values)
    return df


def feature_encoding(df):
    # categorical data encoding
    df = df[df['workclass'] != '?']
    df = df[df['occupation'] != '?']
    df = df[df['native.country'] != '?']
    one_hot_list = ['sex', 'race']
    hash_list = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'native.country']
    ohe = ce.OneHotEncoder(cols=one_hot_list)
    le = ce.OrdinalEncoder()
    he = ce.HashingEncoder(cols=hash_list, drop_invariant=True, n_components=6)
    df['income'] = le.fit_transform(df['income'])
    df = ohe.fit_transform(df)
    df = he.fit_transform(df)
    return df


def initial(df, show_correlation=False, show_histogram=False, show_boxplot=False):
    if show_correlation:
        correlation = df.corr()
        seaborn.heatmap(correlation, xticklabels=df.columns[:-1], yticklabels=df.columns[:-1], annot=True)
        plt.show()
    if show_histogram:
        for column in df.columns[:-1]:
            seaborn.distplot(df[column], color='b')
            plt.title(str(column) + ' histogram')
            plt.show()
    if show_boxplot:
        for column in df.columns[:-1]:
            seaborn.boxplot(df[column])
            plt.title(str(column) + ' boxplot')
            plt.show()


def scale_split(df):
    # Scale data with Standard scaler and split it into train\test subsets
    income_series = df['income']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[df.columns[:-1]])
    scaled_data = pd.DataFrame(data=scaled_data, columns=df.columns[:-1])
    df = scaled_data
    df['income'] = income_series
    return df, RepeatedKFold(n_splits=5, n_repeats=2)


def logistic_regression():
    estimator = LogisticRegression(max_iter=1000)
    parameters = [
        {'solver': ['liblinear'], 'penalty': ['l1'], 'C': [1, 0.1, 0.01]},
        {'solver': ['lbfgs'], 'penalty': ['l2'], 'C': [1, 0.1, 0.01]}]
    logistic_regression_tuple = (estimator, parameters)
    return logistic_regression_tuple


def svm():
    estimator = SVC()
    parameters = [{'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 10]},
                  {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]}]
    svm_tuple = (estimator, parameters)
    return svm_tuple


def knn():
    estimator = KNeighborsClassifier()
    parameters = {'n_neighbors': list(range(2, 20))}
    knn_tuple = (estimator, parameters)
    return knn_tuple


def decision_tree():
    estimator = DecisionTreeClassifier()
    parameters = {'criterion': ['gini', 'entropy'], 'max_depth': list(range(2, 20))}
    decision_tree_tuple = (estimator, parameters)
    return decision_tree_tuple


def random_forest():
    estimator = RandomForestClassifier()
    parameters = {'n_estimators': [100, 150, 200, 300],
                  'criterion': ['gini', 'entropy'], 'max_depth': list(range(2, 20))}
    random_forest_tuple = (estimator, parameters)
    return random_forest_tuple


def grid_search(df, estimator, parameters):
    # Searching over specified parameter values for each estimator
    clf = GridSearchCV(estimator, param_grid=parameters, scoring='roc_auc',
                       refit=True, cv=rkf, iid=False)
    clf.fit(df.iloc[:, :-1], df.iloc[:, -1])
    return clf.best_score_


if __name__ == '__main__':
    # data = preprocessed('adult.csv')
    # data = feature_encoding(data)
    # data.to_csv('adult_encoded.csv', index=False)
    data = pd.read_csv('adult_encoded.csv')
    initial(data)
    data, rkf = scale_split(data)

    # Model comparison
    classifier_list = []
    logistic_regression_classifier = logistic_regression()
    svm_classifier = svm()
    knn_classifier = knn()
    decision_tree_classifier = decision_tree()
    random_forest_classifier = random_forest()
    classifier_list.extend([('Logistic regression', logistic_regression_classifier), ('KNN', knn_classifier),
                            ('Decision tree', decision_tree_classifier), ('SVM', svm_classifier),
                            ('Random forest', random_forest_classifier)])
    for classifier in classifier_list:
        best_score = grid_search(data, classifier[1][0], classifier[1][1])
        print(classifier[0], 'AUC score', best_score)
