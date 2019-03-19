import pandas as pd
import numpy as np
import time
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import matplotlib
import pyswarms as ps

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras import optimizers
from pyswarms.utils.plotters import plot_cost_history

matplotlib.style.use('ggplot')


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)



#Part 1:
# Load and divide dataset
url_dataset = "https://raw.githubusercontent.com/lzs-raja/GA_01/master/dataset/Time-sequence%20datasets/SP500.csv"
dataset = pd.read_csv(url_dataset,encoding = 'utf-8')
x_train = dataset.drop(columns = 'Date')
value_mean = x_train.mean(axis=0)[1]
value_sd = np.sqrt(x_train.var(axis=0)[1])
for i in range(4):
    x_train.iloc[:,i] -= value_mean
    x_train.iloc[:,i] /= value_sd
x_train.iloc[:,4] -= x_train.mean(axis=0)[4]
x_train.iloc[:,4] /= np.sqrt(x_train.var(axis=0)[4])
x_train.head(5)
sample_size = int(x_train.shape[0]/5)
timesteps = 5
features = x_train.shape[1]
used_samples = sample_size*5
#Use the whole week data to predict highest value of next week
x_train = np.asarray(x_train)[:used_samples,:].reshape((sample_size,timesteps,features))
y_train = x_train.max(1)[1:,1].reshape((sample_size-1,1))
x_train = x_train[:-1,:,:]
#print(x_train.shape,y_train.shape)

# Write Keras model LSTM layers and compile the model using SGD
model = Sequential()
model.add(LSTM(128,input_shape=(timesteps,features)))
model.add(Dense(1))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.summary()
total_param = 68737
# Use model fit command to train the model
t0 = time.time()
result_sgd = model.fit(x_train, y_train, epochs=80, batch_size=7,verbose=0)
t1 = time.time()

print("***************************")
print()
print("LSTM Model")
print ("Time taken to train the model: ", t1 - t0, "secs.")
print("Error:",result_sgd.history['loss'][-1])


# Save or plot error with epochs
plt.plot(result_sgd.history['loss'])
plt.show()

#Part 2:

#I use another PSO package called pyswarms here

model_weights =np.array([model.layers[0].get_weights(),model.layers[1].get_weights()])
shape = np.array([model_weights[0][0].shape,model_weights[0][1].shape,model_weights[0][2].shape,model_weights[1][0].shape,model_weights[1][1].shape])
def func(vector_x):
    init_index = 0
    end_index = 0
    end_index += shape[0][0] * shape[0][1]
    model_weights[0][0] = vector_x[init_index:end_index].reshape(shape[0])
    init_index = end_index
    end_index += shape[1][0]*shape[1][1]
    model_weights[0][1] = vector_x[init_index:end_index].reshape(shape[1])
    init_index = end_index
    end_index += shape[2][0]
    model_weights[0][2] = vector_x[init_index:end_index]
    init_index = end_index
    end_index += shape[3][0] * shape[3][1]
    model_weights[1][0] = vector_x[init_index:end_index].reshape(shape[3])
    init_index = end_index
    end_index += shape[4][0]
    model_weights[1][1] = vector_x[init_index:end_index]
    model.layers[0].set_weights(model_weights[0])
    model.layers[1].set_weights(model_weights[1])
    pso_predict = model.predict(x_train)
    error = np.sum(np.square(pso_predict-y_train))/(sample_size-1)
    return error

def swarm_func(x):
    n_particles = x.shape[0]
    j = [func(x[i]) for i in range(n_particles)]
    return np.array(j)

# initialization
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=80, dimensions=total_param,
                                    options=options)


t2 = time.time()
result_pso = optimizer.optimize(swarm_func,iters = 120)
t3 = time.time()

print("Partical Swarm Optimization")
print ("Time taken to train the model: ", t3 - t2, "secs.")
print("Error:",result_pso[0])
print()
print("***************************")

print()
plot_cost_history(optimizer.cost_history)
plt.show()
# Save or plot error with generations/iterations

# Compare performance matrices with Part 1

##################################################################
