from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import datetime

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
err_logistic = 0
err_linear = 0
err_mlp = 0

cnt = 0
for c in classes:
    cnt += 1

    print(c)

    y = train[c]

    # month = train['Month']
    # day = train['Day']
    # hour = train['Hour']

    timestamp = train['timestamp']

    X = []

    cnt = 0
    ind = 0
    for t in timestamp:

        ind += 1
        print(ind, " / ", len(timestamp))

        date_time = pd.to_datetime(t)

        month = date_time.month
        day = date_time.day
        hour = date_time.hour

        before_60 = train[train['timestamp'] == str(date_time - pd.Timedelta(minutes=60))][c].values
        before_90 = train[train['timestamp'] == str(date_time - pd.Timedelta(minutes=90))][c].values
        before_120 = train[train['timestamp'] == str(date_time - pd.Timedelta(minutes=120))][c].values

        if before_60.size == 0:
            # Find closest time to t - 60
            closest = np.argmin(np.abs(timestamp - str(date_time - pd.Timedelta(minutes=60))))
            

            print(before_60)
            exit(0)

        # if before_60.size == 0 and before_90.size == 0 and before_120.size == 0:
        #     #print("DROP")
        #     cnt += 1
        #     continue

        # before_60 = before_60[0]
        # before_90 = before_90[0]
        # before_120 = before_120[0]

        #print(before_60)

        # month = int(str(t)[5:7])
        # day = int(str(t)[8:10])
        # #print(datetime.datetime.strptime(str(t)[0:10], '%Y-%m-%d').weekday())
        # #day = datetime.datetime.strptime(str(t)[0:10], '%Y-%m-%d').weekday()
        # hour = int(str(t)[11:13])

        X.append([month, day, hour])

    print("Dropped ", cnt, " / ", len(timestamp))

    exit(0)

    # hour_max = np.max(X, axis=0)[2]
    # day_max = np.max(X, axis=0)[1]
    # month_max = np.max(X, axis=0)[0]
    # for i in range(0, len(X)):
    #     X[i][0] = X[i][0] / month_max
    #     X[i][1] = X[i][1] / day_max
    #     X[i][2] = X[i][2] / hour_max

    print(" -> Creating models...")
    model_adaboost = AdaBoostClassifier(n_estimators=100, learning_rate=1)
    model_randomforest = RandomForestClassifier(n_estimators=100)
    model_gradientboost = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    model_logistic = linear_model.LogisticRegression(C=1e5)
    model_linear = linear_model.LinearRegression()
    model_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    print(" -> Fitting models...")
    print("    -> AdaBoost")
    model_adaboost.fit(X, y)
    print("    -> RandomForest")
    model_randomforest.fit(X, y)
    print("    -> GradientBoost")
    model_gradientboost.fit(X, y)
    print("    -> Logistic")
    model_logistic.fit(X, y)
    print("    -> Linear")
    model_linear.fit(X, y)
    print("    -> MLP")
    model_mlp.fit(X, y)

    # Predictions for test set
    y_test = test[c]

    timestamp = test['timestamp']

    X_test = []

    for t in timestamp:

        month = int(str(t)[5:7])
        day = int(str(t)[8:10])
        #day = datetime.datetime.strptime(str(t)[0:10], '%Y-%m-%d').weekday()
        hour = int(str(t)[11:13])

        X_test.append([month, day, hour])

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
    print("    -> Logistic")
    y_pred_logistic = model_logistic.predict(X_test)
    print("    -> Linear")
    y_pred_linear = model_linear.predict(X_test)
    print("    -> MLP")
    y_pred_mlp = model_mlp.predict(X_test)

    print("Mean absolute error:")
    err_adaboost += np.mean(abs(y_test - y_pred_adaboost))
    print(" -> AdaBoost: ", err_adaboost/cnt)
    err_randomforest += np.mean(abs(y_test - y_pred_randomforest))
    print(" -> RandomForest: ", err_randomforest/cnt)
    err_gradientboost += np.mean(abs(y_test - y_pred_gradientboost))
    print(" -> GradientBoost: ", err_gradientboost/cnt)
    err_logistic += np.mean(abs(y_test - y_pred_logistic))
    print(" -> Logistic: ", err_logistic/cnt)
    err_linear += np.mean(abs(y_test - y_pred_linear))
    print(" -> Linear: ", err_linear/cnt)
    err_mlp += np.mean(abs(y_test - y_pred_mlp))
    print(" -> MLP: ", err_mlp/cnt)

err_adaboost /= len(classes)
err_randomforest /= len(classes)
err_gradientboost /= len(classes)
err_logistic /= len(classes)
err_linear /= len(classes)
err_mlp /= len(classes)

print("FIINAL MEAN ABSOLUTE ERROR:")
print(" -> AdaBoost: ", err_adaboost)
print(" -> RandomForest: ", err_randomforest)
print(" -> GradientBoost: ", err_gradientboost)
print(" -> Logistic: ", err_logistic)
print(" -> Linear: ", err_linear)
print(" -> MLP: ", err_mlp)
