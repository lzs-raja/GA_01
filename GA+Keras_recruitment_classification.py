import pandas as pd
import numpy as np
import time
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import matplotlib

from keras.models import Sequential
from keras.layers import Dense, Dropout
from geneticalgs import BinaryGA, RealGA, DiffusionGA, MigrationGA

matplotlib.style.use('ggplot')


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

#read data
url_train = "https://raw.githubusercontent.com/lzs-raja/GA_01/master/dataset/Recruitment_dataset/Train.csv"
url_test = "https://raw.githubusercontent.com/lzs-raja/GA_01/master/dataset/Recruitment_dataset/Test.csv"
data_train = pd.read_csv(url_train,encoding = 'utf-8')
data_test = pd.read_csv(url_test,encoding = 'utf-8')

x_train = data_train.drop(columns = ['id','Y'])
y_train = data_train['Y']
x_test = data_test.drop(columns = ['id'])
y_test = pd.DataFrame(np.zeros(x_test.index.size))

#data pretreatment
factor_number = int(x_train.shape[1])
for i in range(factor_number):
    x_train.iloc[:, i] /= x_train.max(axis=0)[i]

#build neural network model
model = Sequential()
model.add(Dense(50, input_dim=factor_number, activation='sigmoid'))
model.add(Dense(25, activation='sigmoid'))
model.add(Dense(12, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

model.summary()
total_param = 2800

#run the model

t0 = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=40000,verbose=0)
t1 = time.time()

print("***************************")
print()
print("Neural Network Model")
print ("Time taken to train the model: ", t1 - t0, "secs.")

y_predict = model.predict(x_train, batch_size=None, verbose=0, steps=None)
y_predict = y_predict[:, 0]
for i in range(len(y_predict)):
    y_predict[i]=1 if y_predict[i]>0.5 else 0
error = np.sum(np.square(y_predict - y_train))

print("Error:",error)


print()
print("***************************")
print()


#get layer_weights' shape
shape = [[(0,0),(0,0)],[(0,0),(0,0)],[(0,0),(0,0)],[(0,0),(0,0)]]
model_weights =np.array([model.layers[0].get_weights(),model.layers[1].get_weights(),model.layers[2].get_weights(),model.layers[3].get_weights()])
for i in range(4):
    shape[i][0] = model_weights[i][0].shape
    shape[i][1] = model_weights[i][1].shape


#use model to define fitness function

def func(vector_x):
    init_index = 0
    end_index = 0
    for i in range(4):
        end_index += shape[i][0][0] * shape[i][0][1]
        model_weights[i][0] = vector_x[init_index:end_index].reshape(shape[i][0])
        init_index = end_index
        end_index += shape[i][1][0]
        model_weights[i][1] = vector_x[init_index:end_index].reshape(shape[i][1])
        init_index = end_index
        model.layers[i].set_weights(model_weights[i])

    y_ga_predict = model.predict(x_train)
    y_ga_predict = y_ga_predict[:, 0]
    for i in range(len(y_ga_predict)):
        y_ga_predict[i] = 1 if y_ga_predict[i] > 0.5 else 0
    error = np.sum(np.square(y_ga_predict - y_train))

    return error

#parameters

generation_num = 50
population_size = 16
elitism = True
selection = "rank"
tournament_size = None # in case of tournament selection
mut_type = 1
mut_prob = 0.05
cross_type = 1
cross_prob = 0.95
optim = "min" # we need to minimize error
interval = (-1,1)
len_vector_x = total_param

#run GA
sga = RealGA(func, optim=optim, elitism=elitism, selection=selection, mut_type=mut_type, mut_prob=mut_prob, cross_type=cross_type, cross_prob=cross_prob)
sga.init_random_population(population_size, len_vector_x, interval)

t2 = time.time()
fitness_progress = sga.run(generation_num)
t3 = time.time()

print("Genetic Algorithms")
print ("Time taken to train the model: ", t3 - t2, "secs.")
print("Error:",sga.best_solution[1])
print()
print("***************************")

print()
plt.plot(list(range(len(fitness_progress))), fitness_progress, 'o')



