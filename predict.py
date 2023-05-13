from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
import pandas as pd
import numpy as np
from datetime import datetime
import copy

holidays = [
    [8, 2],
    [18, 4],
    [27, 4],
    [2, 5],
    [15, 8],
    [31, 10],
    [1, 11],
]

school_holidays = [
    [27, 4],
    [28, 4],
    [29, 4],
    [30, 4],
    [1, 5],
    [2, 5]
]

train = pd.read_csv('bicikelj_train_processed.csv')
test = pd.read_csv('bicikelj_test_processed.csv')

timestamps = [pd.to_datetime(t) for t in test['timestamp'].values]
train_timestamps = [pd.to_datetime(t) for t in train['timestamp'].values]

all_timestamps = timestamps + train_timestamps

# Create a list of the feature column's names
features = train.columns

# Drop first row
# train = train.iloc[2:,:]

# for n in ("Month", "Day", "Hour"):
#     train[n] = train[n] / train[n].max()


classes = features[7:(len(features))]

class_cnt = 0
for c in classes:
    
    class_cnt += 1

    print(class_cnt, "/", len(classes), " : ", c)

    y = train[c]

    X = train[train.columns[1:4]]

    bikesMinus60 = copy.deepcopy(train["bikesMinus60"]).values
    bikesMinus90 = copy.deepcopy(train["bikesMinus90"]).values
    bikesMinus120 = copy.deepcopy(train["bikesMinus120"]).values

    for i in range(0, len(bikesMinus60)):
        bikesMinus60[i] = train[train["timestamp"] == bikesMinus60[i]][c].values[0]
    
    for i in range(0, len(bikesMinus90)):
        bikesMinus90[i] = train[train["timestamp"] == bikesMinus90[i]][c].values[0]

    for i in range(0, len(bikesMinus120)):
        bikesMinus120[i] = train[train["timestamp"] == bikesMinus120[i]][c].values[0]

    holiday_data = []
    school_holiday_data = []
    for t in train["timestamp"]:
        day = pd.to_datetime(t).day
        month = pd.to_datetime(t).month
        date = [day, month]

        if date in holidays:
            holiday_data.append(1)
        else:
            holiday_data.append(0)

        if date in school_holidays or (month == 6 and day >= 26) or month == 7 or month == 8:
            school_holiday_data.append(1)
        else:
            school_holiday_data.append(0)
            

    X = np.column_stack((X, bikesMinus60, bikesMinus90, bikesMinus120, holiday_data, school_holiday_data))

    # print(np.shape(X))

    # for i in range(0, len(X)):
    #     date = [X[i, 1], X[i, 0]]
    #     if date in holidays:
    #         X[i].insert(len(X[i]), 1)
    #     else:
    #         X[i].insert(len(X[i]), 0)

    #     if date in school_holidays or (date[1] == 6 and date[0] >= 26) or date[1] == 7 or date[1] == 8:
    #         X[i].insert(len(X[i]), 1)
    #     else:
    #         X[i].insert(len(X[i]), 0)

    #     print(X[i])
    #     exit(0)


    # X = np.asarray(X, dtype=np.float64)
    # maxes = X.max(axis=0)
    # X = X / maxes

    

    # for i in range(0, len(X[0])):
    #     X[:, i] = float(X[:, i])
    #     X[:,i] = X[:,i] / X[:,i].max()

    #model = AdaBoostClassifier(n_estimators=100, learning_rate=1)
    #model = RandomForestClassifier(n_estimators=100)
    model = linear_model.LinearRegression()
    # model = linear_model.Ridge(alpha=0.5)

    # Fit the model.
    model.fit(X, y)

    X_test = []

    for t in range(0, len(test)):
        timestamp = test['timestamp'][t]

        #print(timestamp)

        row = test.iloc[t, 1:4].values

        bikesMinus60i = copy.deepcopy(test["bikesMinus60"]).values[t]
        bikesMinus90i = copy.deepcopy(test["bikesMinus90"]).values[t]
        bikesMinus120i = copy.deepcopy(test["bikesMinus120"]).values[t]

        bikesMinus60 = train[train["timestamp"] == bikesMinus60i][c].values
        bikesMinus90 = train[train["timestamp"] == bikesMinus90i][c].values
        bikesMinus120 = train[train["timestamp"] == bikesMinus120i][c].values

        bikesMinus60 = bikesMinus60[0] if len(bikesMinus60) > 0 else test[test["timestamp"] == bikesMinus60i][c].values[0]
        bikesMinus90 = bikesMinus90[0] if len(bikesMinus90) > 0 else test[test["timestamp"] == bikesMinus90i][c].values[0]
        bikesMinus120 = bikesMinus120[0] if len(bikesMinus120) > 0 else test[test["timestamp"] == bikesMinus120i][c].values[0]

        day = pd.to_datetime(timestamp).day
        month = pd.to_datetime(timestamp).month
        date = [day, month]

        if date in holidays:
            holiday_data = 1
        else:
            holiday_data = 0

        if date in school_holidays or (month == 6 and day >= 26) or month == 7 or month == 8:
            school_holiday_data = 1
        else:
            school_holiday_data = 0

        X_test = np.column_stack((row[0], row[1], row[2], bikesMinus60, bikesMinus90, bikesMinus120, holiday_data, school_holiday_data))
        X_test = np.asarray(X_test, dtype=np.float64)

        # X_test = X_test / maxes

        #print(X_test)

        y_pred = model.predict(X_test)
        y_pred = str(y_pred[0]).split('.')[0]
        test[c][t] = int(y_pred)


print("Writing to file...")

now = datetime.now()

test = test.drop('month', axis=1)
test = test.drop('day', axis=1)
test = test.drop('hour', axis=1)
test = test.drop('bikesMinus60', axis=1)
test = test.drop('bikesMinus90', axis=1)
test = test.drop('bikesMinus120', axis=1)

file_name = "bicikelj_out_" + str(now.day) + "_" + str(now.month) + "_"  + str(now.hour) + "_" + str(now.minute) + ".csv";
test.to_csv(file_name, index=False)
print("Done!")

    
