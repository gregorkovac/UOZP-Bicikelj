import pandas as pd
import numpy as np
import datetime

test_dataset = pd.read_csv('bicikelj_test.csv')
train_dataset = pd.read_csv('bicikelj_train.csv')

#test = pd.read_csv('bicikelj_test.csv')
features = test_dataset.columns
classes = features[1:(len(features))]

test_timestamps = [pd.to_datetime(t) for t in test_dataset['timestamp'].values]
train_timestamps = [pd.to_datetime(t) for t in train_dataset['timestamp'].values]

all_timestamps = test_timestamps + train_timestamps
#timestamps = np.array(timestamps)

new_data = []

cnt = 0
for t in train_timestamps:
    cnt += 1
    print(cnt, " / ", len(test_timestamps))

    month = t.month
    day = t.day
    hour = t.hour

    bikesMinus60 = str(min(all_timestamps, key=lambda x: abs(x - (t - datetime.timedelta(minutes=60)))))
    bikesMinus90 = str(min(all_timestamps, key=lambda x: abs(x - (t - datetime.timedelta(minutes=90)))))
    bikesMinus120 = str(min(all_timestamps, key=lambda x: abs(x - (t - datetime.timedelta(minutes=120)))))

    row = [str(t), month, day, hour, bikesMinus60, bikesMinus90, bikesMinus120]

    c = test_dataset[test_dataset['timestamp'] == str(t)][classes].values[0]

    for i in range(0, len(c)):
        row.append(c[i])

    #print(row)

    #exit(0)

    # for c in classes:
    #     row.append(dataset[dataset['timestamp'] == str(t)][c].values[0])

    new_data.append(row)

# TODO: Normalize data

# Save new data to csv
new_data = pd.DataFrame(new_data)

# Rename columns
new_data.columns = ['timestamp', 'month', 'day', 'hour', 'bikesMinus60', 'bikesMinus90', 'bikesMinus120'] + classes.tolist()

new_data.to_csv('bicikelj_test_processed.csv', index=False)