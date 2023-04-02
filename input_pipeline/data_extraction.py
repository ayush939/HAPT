import os
import numpy as np
import pandas as pd

# z-score feature normalization
def z_score(df):
    """Z-score normalization"""
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in df_std.columns[:-1]:
        df_std[column] = (df_std[column] - df_std[column].mean()
                          ) / df_std[column].std()

    return df_std

# extracting data from the raw files
def extract_data(mypath, start, end):
    """This function extracts the data from raw files and returns sensor data dataframe and label dataframe (6 parameters + 1 label).
    This function drops the unlabelled data and also returns the normalized features and labels"""

    data = np.empty((1, 7))
    label = []
    labels = np.loadtxt(os.path.join(mypath, 'labels.txt'),
                        delimiter=' ').astype(np.int32)
    if start > 10:
        exp = start*2 - 1
    else:
        exp = 0
    for user in range(start, end+1):
        i = 0
        flag = 0
        while i < 2:
            exp += 1
            i += 1
            acc = np.loadtxt(
                mypath + f'acc_exp{exp:02d}_user{user:02d}.txt', delimiter=' ').astype(np.float32)
            gyro = np.loadtxt(
                mypath + f'gyro_exp{exp:02d}_user{user:02d}.txt', delimiter=' ').astype(np.float32)
            rows = labels[np.where((labels[:, 0] == exp)
                                   * (labels[:, 1] == user))]
            # print(rows)
            rows = rows[rows[:, 3].argsort()]
            for row in rows:
                acceleration = acc[row[3]: row[4]+1, :]
                velocity = gyro[row[3]: row[4]+1, :]
                label = np.ones((acceleration.shape[0], 1))*int(row[2])
                temp_Arr = np.hstack((acceleration, velocity))
                temp_Arr = np.append(temp_Arr, label, axis=1)
                data = np.vstack((data, temp_Arr))

            if user == 10 and i == 2 and flag == 0:
                i = 1
                flag = 1

    data = np.delete(data, 0, 0)
    data = pd.DataFrame({"a_x": data[:, 0],
                        "a_y": data[:, 1],
                         "a_z": data[:, 2],
                         "g_x": data[:, 3],
                         "g_y": data[:, 4],
                         "g_z": data[:, 5],
                         "label": data[:, 6]})
    data = z_score(data)

    return data[["a_x", "a_y", "a_z", "g_x", "g_y", "g_z"]], data[["label"]]
