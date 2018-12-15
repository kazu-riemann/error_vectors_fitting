import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#__ load ECG dataset and standardize it
def load_ecg_data(path):
    #.. load ECG dataset
    df = pd.read_csv(path, header=None, delimiter='\t')
    ecg = df.iloc[:,2].values
    ecg = ecg.reshape(len(ecg), -1)

    #.. standardize data
    scaler = StandardScaler()
    std_ecg = scaler.fit_transform(ecg)

    return std_ecg

#__ plot ECG data up to time 5000
def plot_ecg_data_upto5000(data):
    plt.style.use('ggplot')
    plt.figure(figsize=(15,5))
    plt.xlabel('time')
    plt.ylabel('ECG\'s value')
    plt.plot(np.arange(5000), data[:5000], color='b')
    plt.ylim(-3, 3)
    x = np.arange(4200,4400)
    y1 = [-3]*len(x)
    y2 = [3]*len(x)
    plt.fill_between(x, y1, y2, facecolor='g', alpha=.3)
    plt.savefig("ecg_upto5000.png")
    # plt.show()

#__ plot ECG data after time 5000
def plot_ecg_data_after5000(normal_cycle):
    plt.figure(figsize=(10,5))
    plt.title("training data")
    plt.xlabel('time')
    plt.ylabel('ECG\'s value')
    # stop plot at 8000 times for friendly visual
    plt.plot(np.arange(5000,8000), normal_cycle[:3000], color='b')
    plt.savefig("ecg_after5000.png")
    # plt.show()

#__ create data of the "look_back" length from time-series "ts"
#__ use the next "pred_length" values as labels
def create_subseq(ts, look_back, pred_length):
    sub_seq, next_values = [], []
    for i in range(len(ts)-look_back-pred_length):  
        sub_seq.append(ts[i:i+look_back])
        next_values.append(ts[i+look_back:i+look_back+pred_length].T[0])
    sub_seq = np.array(sub_seq)
    next_values = np.array(next_values)

    return sub_seq, next_values

#__ create LSTM model
def LSTM_model(input_size, look_back, pred_length):
    model = Sequential()
    hidden_unit_size = 35

    #.. use only 1 LSTM layer in this calculation
    model.add(LSTM(hidden_unit_size, input_shape=(input_size, look_back)))
    model.add(Dense(pred_length))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

#__ calculate mean vector and covariance matrix 
def mean_cov(vectors):
    mean = sum(vectors) / len(vectors)
    cov = 0
    for v in vectors:
        cov += np.dot((v - mean).reshape(len(v),1), (v - mean).reshape(1, len(v)))
    cov /= len(vectors)

    print('mean = ', mean)
    print('covariance matrix = ', cov)

    return mean, cov

#__ calculate Mahalanobis distance for a vector "x"
def Mahala_distantce(x,mean,cov):
    d = np.dot(x-mean,np.linalg.inv(cov))
    d = np.dot(d, (x-mean).T)

    return d

#__ plot the original data and
#__ the corresponding Mahalanobis distance sequence
def plot_result(original_data_seq, mahala_distance_seq):
    fig, axes = plt.subplots(nrows=2, figsize=(15,10))

    #.. plot the original data
    axes[0].plot(original_data_seq,color='b',label='original data')
    axes[0].set_xlabel('time')
    axes[0].set_ylabel('ECG\'s value' )
    axes[0].set_ylim(-3, 3)
    x = np.arange(4200,4400)
    y1 = [-3]*len(x)
    y2 = [3]*len(x)
    axes[0].fill_between(x, y1, y2, facecolor='g', alpha=.3)

    #.. plot the corresponding Mahalanobis distance sequence
    axes[1].plot(m_dist, color='r',label='Mahalanobis Distance')
    axes[1].set_xlabel('time')
    axes[1].set_ylabel('Mahalanobis Distance')
    axes[1].set_ylim(0, 1000)
    y1 = [0]*len(x)
    y2 = [1000]*len(x)
    axes[1].fill_between(x, y1, y2, facecolor='g', alpha=.3)

    plt.legend(fontsize=15)
    plt.savefig('result.png')
    # plt.show()

if __name__ == '__main__':
    #.. load ecg data and standardized it
    std_ecg = load_ecg_data('data/qtdbsel102.txt')
    # plot_ecg_data_upto5000(std_ecg)
    upto5000 = std_ecg[:5000]
    normal_cycle = std_ecg[5000:]
    # plot_ecg_data_after5000(normal_cycle)

    look_back = 10
    pred_length = 3

    #.. create training data and test data
    sub_seq, next_values = \
            create_subseq(normal_cycle, look_back, pred_length)
    X_train, X_test, y_train, y_test = \
            train_test_split(sub_seq, next_values, test_size=0.2)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('train size:{}, test size:{}'.format(train_size, test_size))

    #.. reshape input data to be [samples, time steps, features]
    X_train = np.reshape(X_train, (train_size, 1, X_train.shape[1]))

    #.. create model and training it
    model = LSTM_model(1, look_back, pred_length)
    model.fit(X_train, y_train, epochs=100, \
              batch_size=None, shuffle=None, verbose=2)

    #.. make prediction
    X_test = np.reshape(X_test, (test_size, 1, X_test.shape[1]))
    testPredict = model.predict(X_test)

    #.. get mean vector and covariance matrix
    errors = y_test - testPredict
    mean, cov = mean_cov(errors)

    #.. implement anomaly detection to get error vectors
    sub_seq, next_values = \
            create_subseq(upto5000, look_back, pred_length)
    sub_seq = np.reshape(sub_seq, (sub_seq.shape[0], 1, sub_seq.shape[1]))
    pred = model.predict(sub_seq)
    errors = next_values - pred

    #.. get Mahalanobis distance sequence of error vectors
    m_dist = np.zeros(look_back)
    for e in errors:
        m_dist = np.append(m_dist, Mahala_distantce(e,mean,cov))

    #.. plot original data and the result
    plot_result(std_ecg[:5000], m_dist)
