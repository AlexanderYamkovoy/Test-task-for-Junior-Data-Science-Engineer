# Test task for Junior Data Science Engineer
The repository contains an attempt to build a prediction whether income exceeded $50k\year based on census data.
- Task: binary classification;
- Algorithmts to use: Logistic regression, SVM, KNN, Decision Tree (CART), Random Forest;
- Performance measure: ROC AUC score.

The data were scaled by Standard Scaler. The categorical features were encoded with Ordinal, OneHot and Hashing encoders. The best model parameters were tuned by GridSearchCV.

Here is the comparison of the models performance:
- Logistic regression AUC score 0.888;
- KNN AUC score 0.881;
- Decision tree AUC score 0.892;
- SVM AUC score 0.889;
- Random forest AUC score 0.901;

Decision tree, KNN and Logistic regression work faster than SVM and Random Forest, the best performance evaluation have Random forest and Decision tree.
