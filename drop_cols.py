import pandas as pd
import numpy as np
from datetime import datetime
import copy

test = pd.read_csv('bicikelj_out_11_5_11_30.csv')

test = test.drop('month', axis=1)
test = test.drop('day', axis=1)
test = test.drop('hour', axis=1)
test = test.drop('bikesMinus60', axis=1)
test = test.drop('bikesMinus90', axis=1)
test = test.drop('bikesMinus120', axis=1)

now = datetime.now()

file_name = "bicikelj_out_" + str(now.day) + "_" + str(now.month) + "_"  + str(now.hour) + "_" + str(now.minute) + ".csv";
test.to_csv(file_name, index=False)
print("Done!")