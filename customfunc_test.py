
import numpy as np
max_level=64
num_cell=4
num_level=np.exp(np.log(max_level)/num_cell)
print(num_level)
print(np.log(2.78))

info_splited={"num_cell"=num_cell,"num_level",num_level}
print(info_splited)