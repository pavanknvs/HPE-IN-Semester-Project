import matplotlib.pyplot as plt
import numpy as np
np.random.seed(2)
import pandas as pd
import tensorflow as tf
tf.random.set_seed(2)
from sklearn.model_selection import train_test_split


df = pd.read_csv('/content/newtonraphson_dataset.csv')
x= df[['a','b','c']]
y = df[['root']]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.088492, shuffle= True)

model = tf.keras.models.Sequential([
                  tf.keras.layers.Dense(25,input_dim=3,activation='relu'),
                  tf.keras.layers.Dense(40,activation='relu'),
                  tf.keras.layers.Dense(30,activation='relu'),
                  tf.keras.layers.Dense(25,activation='relu'),
                  tf.keras.layers.Dense(1,activation='linear')
                  ])
model.compile(optimizer='adam',loss='mean_squared_error',metrics=[tf.keras.losses.MeanAbsolutePercentageError(),tf.keras.metrics.RootMeanSquaredError()])
csv_logger = tf.keras.callbacks.CSVLogger('NR_CSVLogger.csv', append=False, separator=' ')
hist_2=model.fit(x_train,y_train,batch_size=1024, epochs=1900,validation_split=0.02913,callbacks=[csv_logger])

k=model.predict(x_test)
np.savetxt('/content/NR_Y_observed.csv',k,delimiter=',',fmt='%1.3f')
np.savetxt('/content/NR_Y_original.csv',y_test,delimiter=',',fmt='%1.3f')


plt.plot(hist_2.history['root_mean_squared_error'])
plt.title('Training loss')
plt.ylabel('RMSE Loss')
plt.xlabel('Epoch')
plt.legend([ 'train'], loc='upper right')
plt.ylim(top=10, bottom=0)
plt.show()

plt.plot(hist_2.history['loss'])
plt.title('Training loss')
plt.ylabel('MSE Loss')
plt.xlabel('Epoch')
plt.legend([ 'train'], loc='upper right')
plt.ylim(top=10, bottom=0)
plt.show()

loss = model.evaluate(x_test, y_test)
print(loss)