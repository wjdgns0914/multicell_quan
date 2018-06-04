from datetime import datetime
from scipy.optimize import minimize
import numpy as np
import tensorflow as tf
from contextlib import contextmanager
import sys, os
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def beautifultime():
    timestr = [str(x) for x in list(datetime.now().timetuple())[:6]]
    timestr[1] = timestr[1].rjust(2, '0')
    timestr[2] = timestr[2].rjust(2, '0')
    day = '-'.join(timestr[:3])
    time = '-'.join(timestr[3:])
    #time = '[' + ':'.join(timestr[3:]) + ']'

    return day,time

def my_func(cost_func,x=None):   #20180226  미완성임
    if x is None:
        x=np.random.standard_normal([2,])
    result=minimize(cost_func, x, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
    result=np.float32(result.x)
    return result

def get_distrib(param = [0,0,0,0],size = [1,]):
    assert (len(param) == 4) and (type(param)==type([])), 'The list of parameter should be [Meanmean,Meanstd,Stdmean,Stdstd]'
    meanvalue = np.random.normal(loc=param[0], scale=param[1], size=size)
    stdvalue = ((meanvalue - param[0]) / param[1]) * param[3] + param[2] if param[1]!=0 else np.random.normal(loc=param[2], scale=param[3], size=size)
    fluc_value = tf.cast(tf.reshape(tf.distributions.Normal(loc=meanvalue, scale=stdvalue).sample(1), shape=size),dtype=tf.float32)
    return fluc_value

##파라미터 개수를 세는 용도의 함수
def count_params(var_list):
    num = 0
    for var in var_list:
        if var is not None:
            num += var.get_shape().num_elements()
    return num

#동시에 콘솔과 특정 파일에 print하기위한 함수이다.
def magic_print(*args,file=None):
    print(*args)
    if file!=None:
        print("'''",file=file)
        print(*args, file=file)
        print("'''",file=file)

def staircase(x,ref):
    leng=len(ref)
    y=tf.cond(x<ref[0],lambda: 1,lambda: 0)
    for i in range(leng-1):
        y+=tf.cond(tf.logical_and(tf.greater_equal(x,ref[i]),tf.less(x,ref[i+1])),lambda: i+2, lambda: 0)
    y+=tf.cond(x>ref[-1],lambda: leng+1,lambda: 0)
    return tf.to_float(y)

def make_index(x,ref):
    # leng = len(ref)
    # weight_shape = x.get_shape().as_list()
    # index = tf.ones(shape=weight_shape, dtype=tf.int32)
    # for i in range(leng):
    #     level = tf.ones(shape=weight_shape, dtype=tf.int32) * (leng - i + 1)
    #     mask = tf.logical_and(x>=ref[leng-i-1],tf.equal(index,1))
    #     index = tf.where(mask,level,index)

    # case 2
    # weight_shape = x.get_shape().as_list()
    # reshaped = tf.reshape(x, [weight_shape[0], -1])
    # new_weight_shape= reshaped.get_shape().as_list()
    # index=tf.Variable(initial_value=tf.zeros(shape=new_weight_shape,dtype=tf.float32),trainable=False)
    # for i in range(new_weight_shape[0]):
    #     for j in range(new_weight_shape[1]):
    #         y_one=staircase(reshaped[i,j],ref)
    #         op = tf.assign(index[i, j], y_one)
    #         tf.add_to_collection('op', op)
    # op_list = tf.get_collection('op')
    # with tf.control_dependencies(op_list):
    #     index = tf.identity(index)
    # case 3
    index = tf.to_float(x>=ref[0])
    for ref in ref[1:]:
        index += tf.to_float(x>=ref)
    return index

#20180529: 레벨을 어러 조각으로 나눠주는 함수
# Mode:
# 0:orginal: n진법
# 1:simplified: 단순히 n조각으로 나눠주는 것
# level : index of the level
# max_level : maximum of the level index
# num_level : the number of the level we use per cell
# num_cell  : the number of the cell we use per weight
# n 레벨의 셀을 k개 쓴다고 해보자,
def split_level(level,array_shape, num_cell=None,num_level=None,mode=0):
    assert num_cell!=None and num_level!=None,"Please input num_cell and num_level"
    splited_level = tf.Variable(tf.transpose(tf.zeros(shape=array_shape)),dtype=tf.float32,trainable=False)
    level = tf.transpose(tf.cast(level,dtype=tf.float32))
    if mode==0:
        quotient=level  #level-1 if level=1,2,3,4,5,6...
        for i in range(num_cell):
            assign_splited=tf.assign(splited_level[num_cell - i - 1,:], tf.cast(quotient%num_level,dtype=tf.float32))
            with tf.control_dependencies([assign_splited]):
                quotient = (quotient - splited_level[num_cell - i - 1, :]) / num_level
        with tf.control_dependencies([quotient]):
            splited_level=tf.identity(splited_level)
        # print("Original mode")
    elif mode==1:
        # print("Simplified mode")
        raise NotImplementedError
    else:
        raise ValueError("Error! Please select correct mode.")
    return tf.to_int32(tf.transpose(splited_level))

def convert_level(splited_level, filter_shape, num_cell=None,num_level=None,mode=0):
    assert num_cell != None and num_level != None, "Please input num_cell and num_level"
    bit_weight=num_level**np.arange(num_cell-1,-1,-1)
    if mode==0:
        converted_level=tf.reduce_sum(splited_level*bit_weight,axis=-1)
    elif mode==1:
        raise NotImplementedError
    else:
        raise ValueError("Error! Please select correct mode.")
    return tf.to_int32(converted_level)