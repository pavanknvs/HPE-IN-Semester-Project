import matplotlib.pyplot as plt
import numpy as np
np.random.seed(2)
import pandas as pd
import tensorflow as tf
tf.random.set_seed(2)
from sklearn.model_selection import train_test_split


df = pd.read_csv('/content/LJPotentialDataset.csv')
x= df[['x','y','z']]
y = df[['tf']]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, shuffle= True)

model = tf.keras.models.Sequential([                
                  tf.keras.layers.Dense(15,input_dim=3,activation='relu'),
                  tf.keras.layers.Dense(45,activation='relu'),
                  tf.keras.layers.Dense(15,activation='relu'),
                  tf.keras.layers.Dense(1,activation='linear')
                  ])
model.compile(optimizer='adam',loss='mean_squared_error',metrics=[tf.keras.losses.MeanAbsoluteError(),tf.keras.metrics.RootMeanSquaredError()])
csv_logger = tf.keras.callbacks.CSVLogger('LJ_CSVLogger.csv', append=False, separator=' ')
hist_2=model.fit(x_train,y_train,batch_size=512, epochs=4700,validation_split=0.2,callbacks=[csv_logger])

k=model.predict(x_test)
np.savetxt('/content/LJ_Y_Predicted.csv',k,delimiter=',',fmt='%1.3f')
np.savetxt('/content/LJ_Y_Original.csv',y_test,delimiter=',',fmt='%1.3f')


plt.plot(hist_2.history['val_root_mean_squared_error'])
plt.title('LJ potential Model validity loss')
plt.ylabel('RMSE Loss')
plt.xlabel('Epoch')
plt.legend([ 'train'], loc='upper right')
plt.ylim(top=100, bottom=0)
plt.show()

plt.plot(hist_2.history['loss'])
plt.title('LJ potential model Training loss')
plt.ylabel('MSE Loss')
plt.xlabel('Epoch')
plt.legend([ 'train'], loc='upper right')
plt.ylim(top=100, bottom=0)
plt.show()

plt.plot(hist_2.history['mean_absolute_error'])
plt.title('LJ Potential model Training loss')
plt.ylabel('MAE Loss')
plt.xlabel('Epoch')
plt.legend([ 'train'], loc='upper right')
plt.ylim(top=100, bottom=0)
plt.show()

loss = model.evaluate(x_test, y_test)
print(loss)