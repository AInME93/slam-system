import numpy as np
from nptyping import NDArray
from enum import Enum

def set_np_arr(shape:tuple, values:list) -> NDArray:

    if shape[0]*shape[1] != len(values):
        print('Error')

    np_arr = np.full(shape, 1.0)

    i = 0

    for x in range(shape[0]):
        for y in range(shape[1]):
            np_arr[x,y] = float(values[i])
            i+=1
    
    return np_arr


def set_key_value(file_dir:str) -> dict:

    items = {}

    with open(file_dir, 'r') as f:
        for line in f:
            try:
                k,v = line.strip().split(':')
            except ValueError:
                k = line.strip().split(':')[0]
                v = ":".join(line.strip().split(':')[1:-1])
                
            items[k] = v

    return items

class ExtractType(Enum):
    FAST = 1
    ORB = 2