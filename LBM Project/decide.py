import argparse
import numpy as np
from CPU import *
from GPUkernals import *
from numba import cuda

def main():


    t0 = time.time()
    n = 100
    m = 100

    rhoo = 6.00

    feq = np.full((9, n + 1, m + 1), 0, float)
    f = np.full((9, n + 1, m + 1), 0, float)
    rho = np.full((n + 1, m + 1), rhoo, float)
    w = (4. / 9, 1. / 9, 1. / 9, 1. / 9, 1. / 9, 1. / 36, 1. / 36, 1. / 36, 1. / 36)
    cx = (0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0)
    cy = (0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0)
    u = np.full((n + 1, m + 1), 0, float)
    v = np.full((n + 1, m + 1), 0, float)
    g = np.full((9, n + 1, m + 1), 0, float)
    geq = np.full((9, n + 1, m + 1), 0, float)
    th = np.full((n + 1, m + 1), 0, float)
    uo = 0.0
    sumvelo = 0.0

    dx = 1.0
    dy = dx
    dt = 1.0
    tw = 1.0
    th = np.full((n + 1, m + 1), 0, float)
    ra = 1.0e5
    pr = 0.71
    visco = 0.02
    alpha = visco / pr
    pr = visco / alpha
    gbeta = ra * visco * alpha / (float(m * m * m))
    Re = uo * m / alpha
    omega = 1.0 / (3. * visco + 0.5)
    parameters = [omega, gbeta]

    omegat = 1.0 / (3. * alpha + 0.5)
    timestep = 15000
    t01 = time.time()

    def run_on_cpu(target_function):
        def wrapper (*args, **kwargs):
            startTime = time.time()
            target_function(*args, **kwargs)
            endTime = time.time()
            msecs = (endTime - startTime)*1000
            print("this step used %d ms" % msecs)

        return wrapper


    def run_on_gpu(target_function):
        """compute the appropriate Thread allocation method for target gpu"""
        def dispatch(m,n):
            threadsperblock = (16, 16)
            blockspergrid_x = np.math.ceil(m / threadsperblock[0])
            blockspergrid_y = np.math.ceil(n / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            return threadsperblock, blockspergrid
        TPB,BPG=dispatch(m,n)

        """load the parameters to gpu global memory"""
        """1.Figure out where are the parameters stored"""
        """2.If stored on cpu then copy to gpu with a new name(+'_global')"""

        def parameter_loading(*args,**kwargs):
            for key in args:
                """How to judge if the parameters is existing on GPU or not"""
                locals()[str(key)+'_global']=cuda.to_device(key)


        def wrapper(*args, **kwargs):
            startTime = time.time()
            parameter_loading(*args, **kwargs)
            target_function[TPB,BPG](*args, **kwargs)
            endTime = time.time()
            msecs = (endTime - startTime) * 1000
            print("this step used %d ms" % msecs)

        return wrapper




