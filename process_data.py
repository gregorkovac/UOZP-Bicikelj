import pandas as pd
import numpy as np
from datetime import datetime

test_dataset = pd.read_csv('bicikelj_test.csv')
train_dataset = pd.read_csv('bicikelj_train.csv')

# Join datasets
dataset = pd.concat([train_dataset, test_dataset])

# Sort by timestamp
dataset = dataset.sort_values(by=['timestamp'])

features = dataset.columns
classes = features[1:(len(features))]

timestamps = [pd.to_datetime(t) for t in dataset['timestamp'].values]

new_data = []

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

cnt = 0
for t in timestamps:
    cnt += 1
    print(cnt, " / ", len(timestamps))

    month = t.month
    day = t.day
    hour = t.hour

    bikesMinus60 = str(min(timestamps, key=lambda x: abs(x - (t - pd.Timedelta(minutes=60)))))
    bikesMinus90 = str(min(timestamps, key=lambda x: abs(x - (t - pd.Timedelta(minutes=90)))))
    bikesMinus120 = str(min(timestamps, key=lambda x: abs(x - (t - pd.Timedelta(minutes=120)))))

    date = [day, month]

    if date in holidays:
        holiday_data = 1
    else:
        holiday_data = 0

    if date in school_holidays or (month == 6 and day >= 26) or month == 7 or month == 8:
        school_holiday_data = 1
    else:
        school_holiday_data = 0

    day_of_week = t.weekday()

    row = [str(t), month, day, hour, bikesMinus60, bikesMinus90, bikesMinus120, holiday_data, school_holiday_data, day_of_week]

    c = dataset[dataset['timestamp'] == str(t)][classes].values[0]

    for i in range(0, len(c)):
        row.append(c[i])

    #print(row)

    #exit(0)

    # for c in classes:
    #     row.append(dataset[dataset['timestamp'] == str(t)][c].values[0])

    new_data.append(row)

# Save new data to csv
new_data = pd.DataFrame(new_data)

# Rename columns
new_data.columns = ['timestamp', 'month', 'day', 'hour', 'bikesMinus60', 'bikesMinus90', 'bikesMinus120', 'holidayData', 'schoolHolidayData', 'dayOfWeek'] + classes.tolist()

now = datetime.now()

new_data.to_csv('bicikelj_processed_' + str(now.day) + "_" + str(now.month) + "_"  + str(now.hour) + "_" + str(now.minute) + ".csv", index=False)