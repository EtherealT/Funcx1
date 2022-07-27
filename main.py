# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from funcx.sdk.client import FuncXClient
from train import *
import time
fxc = FuncXClient()
from train import train
from predict import predict
import time

def numba(a, b):
    import numba
    return a + b


def add_func(a, b):
    return a + b


if __name__ == '__main__':
    time1=time.time()
    func_uuid = fxc.register_function(train)
    time2=time.time()
    endpoint1 = '942cf606-bf1c-41bd-a68e-9e35eb2b588e'
    endpoint2 = 'bb05100e-3741-4f62-bde0-5793f4369499'  # Public tutorial endpoint
    res = fxc.run(function_id=func_uuid, endpoint_id=endpoint2)
    time3=time.time()
    while fxc.get_task(res)['pending']:
        time.sleep(3)
    time4=time.time()
    print('function registration time:',time2-time1)
    print('function uploading time:',time3-time2)
    print('function running time: ',time4-time3)
    print(fxc.get_result(res))
    time5=time.time()
    print('result retrieving time:',time5-time4)
