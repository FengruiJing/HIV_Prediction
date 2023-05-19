# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:41:39 2023

@author: FJING
"""

import pandas as pd

# 生成年份数据
years = list(range(2008, 2020))

# 重复数据的次数
repetitions = 2099

# 生成完整的数据集
data = years * repetitions

# 创建DataFrame
df1 = pd.DataFrame({'Year': data})

# 将数据保存为CSV文件
df.to_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/data.csv', index=False)


import pandas as pd

# 读取县的编号数据
county_ids_df = pd.read_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/FIPS.csv')

# 重复次数
repetitions = 12

# 重复县的编号数据
county_ids_repeated = county_ids_df['FIPS'].repeat(repetitions)

# 创建DataFrame
df = pd.DataFrame({'CountyID': county_ids_repeated})

# 将数据保存为CSV文件
df.to_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/CountyID.csv', index=False)


###假设你的原始CSV文件名为 hiv_infection_rates.csv，其中包含年份和各县的HIV感染率数据。以下是将这些数据按照县编号大小整合到一列并保存为新的CSV文件的示例代码：

import pandas as pd

# 读取原始CSV文件
df = pd.read_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/HIVPrevalence_Year.csv')

# 提取县编号列
county_columns = df.columns[1:]  # 假设县编号列从第二列开始

# 创建新的数据列
new_column = []
for county_column in county_columns:
    new_column.extend(df[county_column].tolist())


# 创建新的DataFrame
new_df = pd.DataFrame({'NewHIVPrevalence': new_column})

# 保存为CSV文件
new_df.to_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/HIVPrevalence_YearMerged.csv', index=False)


import pandas as pd

# 读取原始CSV文件 HIVPCI
df = pd.read_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/HIVPCI_Year.csv')

# 提取县编号列
county_columns = df.columns[1:]  # 假设县编号列从第二列开始

# 创建新的数据列
new_column = []
for county_column in county_columns:
    new_column.extend(df[county_column].tolist())


# 创建新的DataFrame
new_df = pd.DataFrame({'NewHIVPCI': new_column})

# 保存为CSV文件
new_df.to_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/HIVPCI_YearMerged.csv', index=False)

import pandas as pd

# 读取原始CSV文件 HIVSCI
df = pd.read_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/HIVSCI_Year.csv')

# 提取县编号列
county_columns = df.columns[1:]  # 假设县编号列从第二列开始

# 创建新的数据列
new_column = []
for county_column in county_columns:
    new_column.extend(df[county_column].tolist())


# 创建新的DataFrame
new_df = pd.DataFrame({'NewHIVSCI': new_column})

# 保存为CSV文件
new_df.to_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/HIVSCI_YearMerged.csv', index=False)

import pandas as pd

# 读取原始CSV文件 HIVDis
df = pd.read_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/HIVDis_Year.csv')

# 提取县编号列
county_columns = df.columns[1:]  # 假设县编号列从第二列开始

# 创建新的数据列
new_column = []
for county_column in county_columns:
    new_column.extend(df[county_column].tolist())


# 创建新的DataFrame
new_df = pd.DataFrame({'NewHIVDis': new_column})

# 保存为CSV文件
new_df.to_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/HIVDis_YearMerged.csv', index=False)




111111111111111111111111111111111111111111111111
111111111111111111111111111111111111111111111111
111111111111111111111111111111111111111111111111
111111111111111111111111111111111111111111111111
111111111111111111111111111111111111111111111111
#2023-05-18 for predicting 2018
#only HIV dignosis

import numpy as np
import pandas as pd
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


# 读取数据
df=pd.read_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/TotalData.csv')

# 将县的编号转换为类别
df['County'] = df['County'].astype('category')

# 初始化预测结果列表
predictions = []

# 处理每个县的数据
for county in df['County'].unique():
    # 提取县的数据
    county_df = df[df['County'] == county]
    
    # 提取训练数据
    train_df = county_df[county_df['Year'] < 2018]

    # 为训练数据进行归一化处理
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train_df[['HIVprevalence']])

    # 准备训练数据和标签
    x_train = np.array([train[i-9:i] for i in range(9, len(train))])
    y_train = np.array([train[i, 0] for i in range(9, len(train))])

    # 调整数据的形状以适应 RNN 模型
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # 构建 RNN 模型
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # 编译和训练 RNN 模型
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=100, verbose=0)

    # 使用 RNN 模型进行预测
    input_data = train[-9:]  # 取最后9个观察值作为输入
    for _ in range(1):  # 预测3年的数据
        input_data = input_data.reshape((1,9, 1))
        pred_single = model.predict(input_data)
        input_data = np.append(input_data[0][1:], [[pred_single[0][0]]], axis=0)
        # 反归一化预测结果并保存
        pred_single = scaler.inverse_transform(pred_single)
        predictions.append((county, 2018+_, pred_single[0][0]))

# 输出预测结果
predictions_df = pd.DataFrame(predictions, columns=['County', 'Year', 'Predicted_HIV_Rate'])
print (predictions_df)
predictions_df.to_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/predictionsHIVprevalence2018_onlyHIVprevalence.csv', index=False)



11111
#  dignose rate +HIVXPCI
import numpy as np
import pandas as pd
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# 设置随机数种子
#1 for only HIV dignose; 2 for HIV+HIVPCI; 3 for HIV+HIV+HIVSCI; 4 for HIV+HIVDistance; 5 for HIV+HIVtotalMOVMENT; 6 for HIV+HIVtotal PCI.
# np.random.seed(5)
# tf.random.set_seed(5)
# random.seed(5)

# np.random.seed(2)
# tf.random.set_seed(2)
# random.seed(2)


# np.random.seed(6)
# tf.random.set_seed(6)
# random.seed(6)

# 读取数据
df=pd.read_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/TotalData.csv')

# 将县的编号转换为类别
df['County'] = df['County'].astype('category')

# 初始化预测结果列表
predictions = []

# 处理每个县的数据
for county in df['County'].unique():
    # 提取县的数据
    county_df = df[df['County'] == county]
    
    # 提取训练数据
    train_df = county_df[county_df['Year'] < 2018]

    # 为训练数据进行归一化处理
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train_df[['HIVprevalence', 'HIVXPCI1']])

    # 准备训练数据和标签
    x_train = np.array([train[i-9:i] for i in range(9, len(train))])
    y_train = np.array([train[i, 0] for i in range(9, len(train))])

    # 调整数据的形状以适应 RNN 模型
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 2))

    # 构建 RNN 模型
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 2)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # 编译和训练 RNN 模型
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=100, verbose=0)

    # 使用 RNN 模型进行预测
    input_data = train[-9:]  # 取最后6个观察值作为输入
    for _ in range(1):  # 预测3年的数据
        input_data = input_data.reshape((1, 9, 2))
        pred_single = model.predict(input_data)
        input_data = np.append(input_data[0][1:], [[pred_single[0][0], train[-1, 1]]], axis=0)
        # 反归一化预测结果并保存
        pred_single = scaler.inverse_transform(np.hstack((pred_single, np.full((1, 1), train[-1, 1]))))[:, 0]
        predictions.append((county, 2018+_, pred_single[0]))

# 输出预测结果
predictions_df = pd.DataFrame(predictions, columns=['County', 'Year', 'Predicted_HIV_Rate'])
print (predictions_df)
predictions_df.to_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/predictionsHIVprevalence2018_onlyHIVprevalenceHIVPCI.csv', index=False)


11111
#  dignose rate +HIVSCI
import numpy as np
import pandas as pd
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# 设置随机数种子
#1 for only HIV dignose; 2 for HIV+HIVPCI; 3 for HIV+HIV+HIVSCI; 4 for HIV+HIVDistance; 5 for HIV+HIVtotalMOVMENT; 6 for HIV+HIVtotal PCI.
# np.random.seed(5)
# tf.random.set_seed(5)
# random.seed(5)

# np.random.seed(2)
# tf.random.set_seed(2)
# random.seed(2)


# np.random.seed(6)
# tf.random.set_seed(6)
# random.seed(6)

# 读取数据
df=pd.read_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/TotalData.csv')

# 将县的编号转换为类别
df['County'] = df['County'].astype('category')

# 初始化预测结果列表
predictions = []

# 处理每个县的数据
for county in df['County'].unique():
    # 提取县的数据
    county_df = df[df['County'] == county]
    
    # 提取训练数据
    train_df = county_df[county_df['Year'] < 2018]

    # 为训练数据进行归一化处理
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train_df[['HIVprevalence', 'HIVSCI1']])

    # 准备训练数据和标签
    x_train = np.array([train[i-9:i] for i in range(9, len(train))])
    y_train = np.array([train[i, 0] for i in range(9, len(train))])

    # 调整数据的形状以适应 RNN 模型
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 2))

    # 构建 RNN 模型
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 2)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # 编译和训练 RNN 模型
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=100, verbose=0)

    # 使用 RNN 模型进行预测
    input_data = train[-9:]  # 取最后6个观察值作为输入
    for _ in range(1):  # 预测3年的数据
        input_data = input_data.reshape((1, 9, 2))
        pred_single = model.predict(input_data)
        input_data = np.append(input_data[0][1:], [[pred_single[0][0], train[-1, 1]]], axis=0)
        # 反归一化预测结果并保存
        pred_single = scaler.inverse_transform(np.hstack((pred_single, np.full((1, 1), train[-1, 1]))))[:, 0]
        predictions.append((county, 2018+_, pred_single[0]))

# 输出预测结果
predictions_df = pd.DataFrame(predictions, columns=['County', 'Year', 'Predicted_HIV_Rate'])
print (predictions_df)
predictions_df.to_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/predictionsHIVprevalence2018_onlyHIVprevalenceHIVSCI.csv', index=False)

11111
#  dignose rate +HIVdistance
import numpy as np
import pandas as pd
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# 设置随机数种子
#1 for only HIV dignose; 2 for HIV+HIVPCI; 3 for HIV+HIV+HIVSCI; 4 for HIV+HIVDistance; 5 for HIV+HIVtotalMOVMENT; 6 for HIV+HIVtotal PCI.
# np.random.seed(5)
# tf.random.set_seed(5)
# random.seed(5)

# np.random.seed(2)
# tf.random.set_seed(2)
# random.seed(2)


# np.random.seed(6)
# tf.random.set_seed(6)
# random.seed(6)

# 读取数据
df=pd.read_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/TotalData.csv')

# 将县的编号转换为类别
df['County'] = df['County'].astype('category')

# 初始化预测结果列表
predictions = []

# 处理每个县的数据
for county in df['County'].unique():
    # 提取县的数据
    county_df = df[df['County'] == county]
    
    # 提取训练数据
    train_df = county_df[county_df['Year'] < 2018]

    # 为训练数据进行归一化处理
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train_df[['HIVprevalence', 'HIVDis1']])

    # 准备训练数据和标签
    x_train = np.array([train[i-9:i] for i in range(9, len(train))])
    y_train = np.array([train[i, 0] for i in range(9, len(train))])

    # 调整数据的形状以适应 RNN 模型
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 2))

    # 构建 RNN 模型
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 2)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # 编译和训练 RNN 模型
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=100, verbose=0)

    # 使用 RNN 模型进行预测
    input_data = train[-9:]  # 取最后6个观察值作为输入
    for _ in range(1):  # 预测3年的数据
        input_data = input_data.reshape((1, 9, 2))
        pred_single = model.predict(input_data)
        input_data = np.append(input_data[0][1:], [[pred_single[0][0], train[-1, 1]]], axis=0)
        # 反归一化预测结果并保存
        pred_single = scaler.inverse_transform(np.hstack((pred_single, np.full((1, 1), train[-1, 1]))))[:, 0]
        predictions.append((county, 2018+_, pred_single[0]))

# 输出预测结果
predictions_df = pd.DataFrame(predictions, columns=['County', 'Year', 'Predicted_HIV_Rate'])
print (predictions_df)
predictions_df.to_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/predictionsHIVprevalence2018_onlyHIVprevalenceHIVDis.csv', index=False)


111111111111111111111111111111111111111111111111
111111111111111111111111111111111111111111111111
111111111111111111111111111111111111111111111111
111111111111111111111111111111111111111111111111
111111111111111111111111111111111111111111111111
#2023-05-18 for predicting 2019
#only HIV dignosis

import numpy as np
import pandas as pd
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


# 读取数据
df=pd.read_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/TotalData.csv')

# 将县的编号转换为类别
df['County'] = df['County'].astype('category')

# 初始化预测结果列表
predictions = []

# 处理每个县的数据
for county in df['County'].unique():
    # 提取县的数据
    county_df = df[df['County'] == county]
    
    # 提取训练数据
    train_df = county_df[county_df['Year'] < 2019]

    # 为训练数据进行归一化处理
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train_df[['HIVprevalence']])

    # 准备训练数据和标签
    x_train = np.array([train[i-10:i] for i in range(10, len(train))])
    y_train = np.array([train[i, 0] for i in range(10, len(train))])

    # 调整数据的形状以适应 RNN 模型
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # 构建 RNN 模型
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # 编译和训练 RNN 模型
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=100, verbose=0)

    # 使用 RNN 模型进行预测
    input_data = train[-10:]  # 取最后9个观察值作为输入
    for _ in range(1):  # 预测3年的数据
        input_data = input_data.reshape((1,10, 1))
        pred_single = model.predict(input_data)
        input_data = np.append(input_data[0][1:], [[pred_single[0][0]]], axis=0)
        # 反归一化预测结果并保存
        pred_single = scaler.inverse_transform(pred_single)
        predictions.append((county, 2019+_, pred_single[0][0]))

# 输出预测结果
predictions_df = pd.DataFrame(predictions, columns=['County', 'Year', 'Predicted_HIV_Rate'])
print (predictions_df)
predictions_df.to_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/predictionsHIVprevalence2019_onlyHIVprevalence.csv', index=False)



11111
#  dignose rate +HIVXPCI
import numpy as np
import pandas as pd
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# 设置随机数种子
#1 for only HIV dignose; 2 for HIV+HIVPCI; 3 for HIV+HIV+HIVSCI; 4 for HIV+HIVDistance; 5 for HIV+HIVtotalMOVMENT; 6 for HIV+HIVtotal PCI.
# np.random.seed(5)
# tf.random.set_seed(5)
# random.seed(5)

# np.random.seed(2)
# tf.random.set_seed(2)
# random.seed(2)


# np.random.seed(6)
# tf.random.set_seed(6)
# random.seed(6)

# 读取数据
df=pd.read_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/TotalData.csv')

# 将县的编号转换为类别
df['County'] = df['County'].astype('category')

# 初始化预测结果列表
predictions = []

# 处理每个县的数据
for county in df['County'].unique():
    # 提取县的数据
    county_df = df[df['County'] == county]
    
    # 提取训练数据
    train_df = county_df[county_df['Year'] < 2019]

    # 为训练数据进行归一化处理
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train_df[['HIVprevalence', 'HIVXPCI1']])

    # 准备训练数据和标签
    x_train = np.array([train[i-10:i] for i in range(10, len(train))])
    y_train = np.array([train[i, 0] for i in range(10, len(train))])

    # 调整数据的形状以适应 RNN 模型
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 2))

    # 构建 RNN 模型
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 2)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # 编译和训练 RNN 模型
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=100, verbose=0)

    # 使用 RNN 模型进行预测
    input_data = train[-10:]  # 取最后6个观察值作为输入
    for _ in range(1):  # 预测3年的数据
        input_data = input_data.reshape((1, 10, 2))
        pred_single = model.predict(input_data)
        input_data = np.append(input_data[0][1:], [[pred_single[0][0], train[-1, 1]]], axis=0)
        # 反归一化预测结果并保存
        pred_single = scaler.inverse_transform(np.hstack((pred_single, np.full((1, 1), train[-1, 1]))))[:, 0]
        predictions.append((county, 2019+_, pred_single[0]))

# 输出预测结果
predictions_df = pd.DataFrame(predictions, columns=['County', 'Year', 'Predicted_HIV_Rate'])
print (predictions_df)
predictions_df.to_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/predictionsHIVprevalence2019_onlyHIVprevalenceHIVPCI.csv', index=False)


11111
#  dignose rate +HIVSCI
import numpy as np
import pandas as pd
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# 设置随机数种子
#1 for only HIV dignose; 2 for HIV+HIVPCI; 3 for HIV+HIV+HIVSCI; 4 for HIV+HIVDistance; 5 for HIV+HIVtotalMOVMENT; 6 for HIV+HIVtotal PCI.
# np.random.seed(5)
# tf.random.set_seed(5)
# random.seed(5)

# np.random.seed(2)
# tf.random.set_seed(2)
# random.seed(2)


# np.random.seed(6)
# tf.random.set_seed(6)
# random.seed(6)

# 读取数据
df=pd.read_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/TotalData.csv')

# 将县的编号转换为类别
df['County'] = df['County'].astype('category')

# 初始化预测结果列表
predictions = []

# 处理每个县的数据
for county in df['County'].unique():
    # 提取县的数据
    county_df = df[df['County'] == county]
    
    # 提取训练数据
    train_df = county_df[county_df['Year'] < 2019]

    # 为训练数据进行归一化处理
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train_df[['HIVprevalence', 'HIVSCI1']])

    # 准备训练数据和标签
    x_train = np.array([train[i-10:i] for i in range(10, len(train))])
    y_train = np.array([train[i, 0] for i in range(10, len(train))])

    # 调整数据的形状以适应 RNN 模型
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 2))

    # 构建 RNN 模型
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 2)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # 编译和训练 RNN 模型
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=100, verbose=0)

    # 使用 RNN 模型进行预测
    input_data = train[-10:]  # 取最后6个观察值作为输入
    for _ in range(1):  # 预测3年的数据
        input_data = input_data.reshape((1, 10, 2))
        pred_single = model.predict(input_data)
        input_data = np.append(input_data[0][1:], [[pred_single[0][0], train[-1, 1]]], axis=0)
        # 反归一化预测结果并保存
        pred_single = scaler.inverse_transform(np.hstack((pred_single, np.full((1, 1), train[-1, 1]))))[:, 0]
        predictions.append((county, 2019+_, pred_single[0]))

# 输出预测结果
predictions_df = pd.DataFrame(predictions, columns=['County', 'Year', 'Predicted_HIV_Rate'])
print (predictions_df)
predictions_df.to_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/predictionsHIVprevalence2019_onlyHIVprevalenceHIVSCI.csv', index=False)

11111
#  dignose rate +HIVdistance
import numpy as np
import pandas as pd
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# 设置随机数种子
#1 for only HIV dignose; 2 for HIV+HIVPCI; 3 for HIV+HIV+HIVSCI; 4 for HIV+HIVDistance; 5 for HIV+HIVtotalMOVMENT; 6 for HIV+HIVtotal PCI.
# np.random.seed(5)
# tf.random.set_seed(5)
# random.seed(5)

# np.random.seed(2)
# tf.random.set_seed(2)
# random.seed(2)


# np.random.seed(6)
# tf.random.set_seed(6)
# random.seed(6)

# 读取数据
df=pd.read_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/TotalData.csv')

# 将县的编号转换为类别
df['County'] = df['County'].astype('category')

# 初始化预测结果列表
predictions = []

# 处理每个县的数据
for county in df['County'].unique():
    # 提取县的数据
    county_df = df[df['County'] == county]
    
    # 提取训练数据
    train_df = county_df[county_df['Year'] < 2019]

    # 为训练数据进行归一化处理
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train_df[['HIVprevalence', 'HIVDis1']])

    # 准备训练数据和标签
    x_train = np.array([train[i-10:i] for i in range(10, len(train))])
    y_train = np.array([train[i, 0] for i in range(10, len(train))])

    # 调整数据的形状以适应 RNN 模型
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 2))

    # 构建 RNN 模型
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 2)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # 编译和训练 RNN 模型
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=100, verbose=0)

    # 使用 RNN 模型进行预测
    input_data = train[-10:]  # 取最后6个观察值作为输入
    for _ in range(1):  # 预测3年的数据
        input_data = input_data.reshape((1, 10, 2))
        pred_single = model.predict(input_data)
        input_data = np.append(input_data[0][1:], [[pred_single[0][0], train[-1, 1]]], axis=0)
        # 反归一化预测结果并保存
        pred_single = scaler.inverse_transform(np.hstack((pred_single, np.full((1, 1), train[-1, 1]))))[:, 0]
        predictions.append((county, 2019+_, pred_single[0]))

# 输出预测结果
predictions_df = pd.DataFrame(predictions, columns=['County', 'Year', 'Predicted_HIV_Rate'])
print (predictions_df)
predictions_df.to_csv('D:/Data/Paper_HIV/FinalExperimentHIVPrevalencePrediction/predictionsHIVprevalence2019_onlyHIVprevalenceHIVDis.csv', index=False)

