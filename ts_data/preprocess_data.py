import pandas
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def transform(arr):
    processor = MinMaxScaler()
    processor.fit(arr)
    return processor.transform(arr)


# read data frame from csv file
data = pandas.read_csv("./data1.csv")

# delete useless dims
del data["1"]
del data["11"]
del data["15"]
del data["18"]

# get numpy 2d-array from the data frame
data_array = data.values

# only keep the sequence segment with no missing observations
arr_train = data_array[170000:470000]
arr_valid = data_array[:60000]
arr_test = data_array[60000:120000]

# normalize the data
arr_train = transform(arr_train)
arr_valid = transform(arr_valid)
arr_test = transform(arr_test)

# save the preprocessed data
np.save('./data/data_train', arr_train)
np.save('./data/data_valid', arr_valid)
np.save('./data/data_test', arr_test)