from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from datetime import datetime

train = pd.read_csv('bicikelj_train_month_day.csv')
test = pd.read_csv('bicikelj_test.csv')

# Create a list of the feature column's names
features = train.columns

# Drop first row
train = train.iloc[2:,:]

# for n in ("Month", "Day", "Hour"):
#     train[n] = train[n] / train[n].max()


classes = features[3:(len(features))]

for c in classes:
    
    print(c)

    y = train[c]

    month = train['Month']
    day = train['Day']
    Hour = train['Hour']

    X = np.column_stack((month, day, Hour))

    # Create the model with 100 trees
    #model = AdaBoostClassifier(n_estimators=100, learning_rate=1)
    model = RandomForestClassifier(n_estimators=100)

    # Fit the model.
    model.fit(X, y)

    for t in range(0, len(test)):
        timestamp = test['timestamp'][t]

        month = int(str(timestamp)[5:7])
        day = int(str(timestamp)[8:10])
        Hour = int(str(timestamp)[11:13])

        X_test = np.column_stack((month, day, Hour))

        y_pred = model.predict(X_test)

        y_pred = str(y_pred[0]).split('.')[0]

        test[c][t] = int(y_pred)

print("Writing to file...")

now = datetime.now()

file_name = "bicikelj_out_" + str(now.day) + "_" + str(now.month) + "_"  + str(now.hour) + "_" + str(now.minute) + ".csv";
test.to_csv(file_name, index=False)
print("Done!")

    
