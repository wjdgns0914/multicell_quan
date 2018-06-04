from funcLayer import *
FLAGS = tf.app.flags.FLAGS
# print("Model")

model = Sequential([
    BinarizedAffine(512, bias=False,name='L1_FullyConnected'),
    ReLU(name='L2_ReLU'),
    BinarizedAffine(10,bias=False,name='L3_FullyConnected'),
])
