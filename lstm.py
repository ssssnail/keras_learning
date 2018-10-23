import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import LSTM

#define the dataset

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#create mapping of characters to integers(0-25)and the reverse

char_to_int = dict((c,i) for i,c in enumerate(alphabet))
int_to_char = dict((i,c) for i,c in enumerate(alphabet))
seq_length = 3
dataX = []
dataY = []
for i in range(0,len(alphabet)-seq_length,1):
    seq_in = alphabet[i:i+seq_length]
    seq_out = alphabet[i+seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append([char_to_int[seq_out]])

X = np.reshape(dataX,(len(dataX),seq_length,1))

#normalize
X = X/float(len(alphabet))

y = np_utils.to_categorical(dataY)
#create and fit model
model = Sequential()
model.add(LSTM(32,input_shape=(X.shape[1],X.shape[2])))
model.add(Dense(y.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#model.fit(X,y,epochs=300,batch_size=1,verbose=2)
scores = model.evaluate(X,y,verbose=0)
print(scores)


for pattern in dataX:
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print (seq_in, "->", result)

