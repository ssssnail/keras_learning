from keras import Model
from keras.layers import Input,Dense,Reshape
from keras import Sequential

a = Input(shape=(32,))
b = Dense(32)(a)

model = Model(inputs=a,outputs=b)
print(model)

layer = Dense(32)
config = layer.get_config()
reconstructed_layer = Dense.get_config(config)

model = Sequential()
model.add(Reshape((3,4),input_shape=(12,)))
#now,model.output_shape  == (None,3,4)
#note: 'None' is the batch dimension

