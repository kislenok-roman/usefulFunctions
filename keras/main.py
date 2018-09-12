# https://www.datacamp.com/community/tutorials/deep-learning-jupyter-aws

# 1. architecture
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical # для создания dummy

predictors = np.loadtxt(filename, delimiter = ",")

model = Sequential() # последовательная модель
# Добавляем слой, Dense -- полностью связанный с предыдущим слой
# 100 - число нейронов слоя
# activation -- задаёт функцию передачи (f): this_layer = f(prev_layer)
#   relu = f(x): max(0, x) -- could lead to "dead neuron" problem (f(-3) == 0 -- no update)
#   tanh - avoid this problem, bu has "horizontal" parts with d' --> 0 (no update) "Vamishing gradients"
# input_shape задаёт формат входных данных -- фиксированное число столбцов и сколько угодно строк (пустой второй) 
model.add(Dense(100, activation = "relu", input_shape = (predictors.shape[1], ))) 
model.add(Dense(100, activation = "relu"))
model.add(Dense(1)) # выходной слой для регрессии

# num of layers and the layer sizes increasy model capacity -- the ability to fit to any data
# so low capacity -- underfitting, high capacity -- overfitting

# model.add(Dense(N, activation = "softmax")) для классификации на N классов
# 2. compile
# optimizer - how the model would optimize
#   "adam":
#     https://keras.io/optimizers/#adam
#     https://arxiv.org/pdf/1412.6980v8.pdf
#   "sgd" -- Stochastic Gradient Descent
#   keras.optimizers.SGD(lr = 0.01) - tuning optimizer
# loss - loss function used to optimize:
#   mean_squared_error - for regression
#   categorical_crossentropy - for classification
# metrics = ["accuracy"] -- необязательный параметр, позволяет следить при выводе эпох за показателем
model.compile(optimizer = "adam", loss = "mean_squared_error")
# 3. fit
# can internaly use split for train/validation
from keras.callbacks import EarlyStopping
# patience -- how many epochs (rounds of putting all data into training) will wait to stop when no improvement
early_stopping_monitor = EarlyStopping(patience=2)
model.fit(predictors, target, validation_split=0.3, epochs = 20, callbacks=[early_stopping_monitor])
# 4. predict
from keras.models import load_model

model.save("filename.h5")
# resore: load_model("filename.h5")

predictions = model.predict(new_data) # for inary classification predictions[:, 1] 
print(model.summary())