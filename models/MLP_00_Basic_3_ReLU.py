from funcLayer import *
FLAGS = tf.app.flags.FLAGS
# print("Model")
# print(FLAGS.Drift1,FLAGS.Drift2)
model = Sequential([
    BinarizedAffine(256, bias=False,name='L1_FullyConnected'),
    ReLU(name='L2_ReLU'),
    BinarizedAffine(256, bias=False,name='L3_FullyConnected'),
    ReLU(name='L4_ReLU'),
    BinarizedAffine(10,bias=False,name='L5_FullyConnected'),
])
# Ver 1.0
"""
model = Sequential([
    BinarizedAffine(256, bias=False,name='L1_FullyConnected'),
    Sigmoid(name='L2_Sigmoid'),
    BinarizedAffine(256, bias=False,name='L3_FullyConnected'),
    Sigmoid(name='L4_Sigmoid'),
    BinarizedAffine(10,bias=False,name='L5_FullyConnected'),
])
"""
