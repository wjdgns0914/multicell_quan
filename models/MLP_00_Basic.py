from funcLayer import *
FLAGS = tf.app.flags.FLAGS
# print("Model")

model = Sequential([
    BinarizedAffine(256, bias=False,name='L1_FullyConnected'),
    Sigmoid(name='L2_Sigmoid'),
    BinarizedAffine(10,bias=False,name='L3_FullyConnected'),
])
