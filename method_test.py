from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
import pandas as pd
import numpy as np
import datetime
import copy

dataset = pd.read_csv('bicikelj_train_processed.csv')
#test = pd.read_csv('bicikelj_test.csv')
features = dataset.columns
classes = features[7:(len(features))]

# Drop first row
#dataset = dataset.iloc[2:,:]

# for f in features:
#     for d in range(2, len(dataset[f])):
#         dataset[f][d] = float(dataset[f][d])

train_mask = np.random.rand(len(dataset)) < 0.8

# Split dataset into training and test sets
train = dataset[train_mask]
test = dataset[~train_mask]

err_adaboost = 0
err_randomforest = 0
err_gradientboost = 0
# err_logistic = 0
err_linear = 0

timestamps = [pd.to_datetime(t) for t in dataset['timestamp'].values]

# print("Preparing test times...")
# test_times = []

# for t in timestamps:
#     t_obj = pd.to_datetime(t)

#     month = t_obj.month
#     day = t_obj.day
#     hour = t_obj.hour

#     bikesMinus60 = str(min(timestamps, key=lambda x: abs(x - (t_obj - datetime.timedelta(minutes=60)))))
#     bikesMinus90 = str(min(timestamps, key=lambda x: abs(x - (t_obj - datetime.timedelta(minutes=90)))))
#     bikesMinus120 = str(min(timestamps, key=lambda x: abs(x - (t_obj - datetime.timedelta(minutes=120)))))

#     test_times.append([t, month, day, hour, bikesMinus60, bikesMinus90, bikesMinus120])

# print(test_times)

cnt = 0
for c in classes:
    cnt += 1

    print(c)

    y = train[c]

    # month = train['Month']
    # day = train['Day']
    # hour = train['Hour']

    print("Preparing train set...")

    X = train[train.columns[1:4]]
    #print(X)

    bikesMinus60 = copy.deepcopy(train["bikesMinus60"]).values
    bikesMinus90 = copy.deepcopy(train["bikesMinus90"]).values
    bikesMinus120 = copy.deepcopy(train["bikesMinus120"]).values

    for i in range(0, len(bikesMinus60)):
        bikesMinus60[i] = dataset[dataset["timestamp"] == bikesMinus60[i]][c].values[0]
    
    for i in range(0, len(bikesMinus90)):
        bikesMinus90[i] = dataset[dataset["timestamp"] == bikesMinus90[i]][c].values[0]

    for i in range(0, len(bikesMinus120)):
        bikesMinus120[i] = dataset[dataset["timestamp"] == bikesMinus120[i]][c].values[0]

    X = np.column_stack((X, bikesMinus60, bikesMinus90, bikesMinus120))

    print(" -> Creating models...")
    model_adaboost = AdaBoostClassifier(n_estimators=100, learning_rate=1)
    model_randomforest = RandomForestClassifier(n_estimators=100)
    model_gradientboost = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    #model_logistic = linear_model.LogisticRegression(C=1e5)
    model_linear = linear_model.LinearRegression()

    print(" -> Fitting models...")
    print("    -> AdaBoost")
    model_adaboost.fit(X, y)
    print("    -> RandomForest")
    model_randomforest.fit(X, y)
    print("    -> GradientBoost")
    model_gradientboost.fit(X, y)
    #print("    -> Logistic")
    #model_logistic.fit(X, y)
    print("    -> Linear")
    model_linear.fit(X, y)

    # Predictions for test set
    y_test = test[c]

    print("Preparing test set...")
    timestamp = test['timestamp']

    X_test = test[test.columns[1:4]]
    #print(X_test)

    bikesMinus60 = copy.deepcopy(test["bikesMinus60"]).values
    bikesMinus90 = copy.deepcopy(test["bikesMinus90"]).values
    bikesMinus120 = copy.deepcopy(test["bikesMinus120"]).values

    for i in range(0, len(bikesMinus60)):
        bikesMinus60[i] = dataset[dataset["timestamp"] == bikesMinus60[i]][c].values[0]
    
    for i in range(0, len(bikesMinus90)):
        bikesMinus90[i] = dataset[dataset["timestamp"] == bikesMinus90[i]][c].values[0]

    for i in range(0, len(bikesMinus120)):
        bikesMinus120[i] = dataset[dataset["timestamp"] == bikesMinus120[i]][c].values[0]

    X_test = np.column_stack((X_test, bikesMinus60, bikesMinus90, bikesMinus120))

    # hour_max = np.max(X_test, axis=0)[2]
    # day_max = np.max(X_test, axis=0)[1]
    # month_max = np.max(X_test, axis=0)[0]
    # for i in range(0, len(X_test)):
    #     X_test[i][0] = X_test[i][0] / month_max
    #     X_test[i][1] = X_test[i][1] / day_max
    #     X_test[i][2] = X_test[i][2] / hour_max

    #X_test = np.column_stack((month, day, hour))

    print(" -> Predicting...")
    print("    -> AdaBoost")
    y_pred_adaboost = model_adaboost.predict(X_test)
    print("    -> RandomForest")
    y_pred_randomforest = model_randomforest.predict(X_test)
    print("    -> GradientBoost")
    y_pred_gradientboost = model_gradientboost.predict(X_test)
    #print("    -> Logistic")
    #y_pred_logistic = model_logistic.predict(X_test)
    print("    -> Linear")
    y_pred_linear = model_linear.predict(X_test)

    print("Mean absolute error:")
    err_adaboost += np.mean(abs(y_test - y_pred_adaboost))
    print(" -> AdaBoost: ", err_adaboost/cnt)
    err_randomforest += np.mean(abs(y_test - y_pred_randomforest))
    print(" -> RandomForest: ", err_randomforest/cnt)
    err_gradientboost += np.mean(abs(y_test - y_pred_gradientboost))
    print(" -> GradientBoost: ", err_gradientboost/cnt)
    #err_logistic += np.mean(abs(y_test - y_pred_logistic))
    #print(" -> Logistic: ", err_logistic/cnt)
    err_linear += np.mean(abs(y_test - y_pred_linear))
    print(" -> Linear: ", err_linear/cnt)

err_adaboost /= len(classes)
err_randomforest /= len(classes)
err_gradientboost /= len(classes)
#err_logistic /= len(classes)
err_linear /= len(classes)

print("FIINAL MEAN ABSOLUTE ERROR:")
print(" -> AdaBoost: ", err_adaboost)
print(" -> RandomForest: ", err_randomforest)
print(" -> GradientBoost: ", err_gradientboost)
# print(" -> Logistic: ", err_logistic)
print(" -> Linear: ", err_linear)
