import numpy as np
from numba import cuda

n=10
m=10

rhoo = 6.00

feq = np.full((9, n + 1, m + 1), 0, np.float)
fin = np.full((9, n + 1, m + 1), 0, np.float)
rho = np.full((n + 1, m + 1), rhoo, np.float)
w = (4. / 9, 1. / 9, 1. / 9, 1. / 9, 1. / 9, 1. / 36, 1. / 36, 1. / 36, 1. / 36)
cx = (0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0)
cy = (0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0)
u = np.full((n + 1, m + 1), 0, np.float)
v = np.full((n + 1, m + 1), 0, np.float)
g = np.full((9, n + 1, m + 1), 0, np.float)
geq = np.full((9, n + 1, m + 1), 0, np.float)
th = np.full((n + 1, m + 1), 0, np.float)
uo = 0.0
sumvelo = 0.0

dx = 1.0
dy = dx
dt = 1.0
tw = 1.0
th = np.full((n + 1, m + 1), 0, np.float)
ra = 1.0e5
pr = 0.71
visco = 0.02
alpha = visco / pr
pr = visco / alpha
gbeta = ra * visco * alpha / (float(m * m * m))
Re = uo * m / alpha
omega = 1.0 / (3. * visco + 0.5)

omegat = 1.0 / (3. * alpha + 0.5)
mstep = 15
for i in range(0, n):
    u[i, m] = uo
    v[i, m] = 0.0

SM = 10

def dispatch(m, n):
    threadsperblock = (16, 16)
    blockspergrid_x = np.math.ceil(m / threadsperblock[0])
    blockspergrid_y = np.math.ceil(n / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    return threadsperblock, blockspergrid

TPB=16
TPB2D, BPG2D = dispatch(n, m)
u_global=cuda.to_device(u)
v_global=cuda.to_device(v)
f_global=cuda.to_device(fin)
feq_global = cuda.to_device(feq)
rho_global = cuda.to_device(rho)

omega_global = cuda.to_device(omega)
w_global = cuda.to_device(w)
cx_global = cuda.to_device(cx)
cy_global = cuda.to_device(cy)
n_global = cuda.to_device(n)
m_global = cuda.to_device(m)
th_global=cuda.to_device(th)
gbeta_global = cuda.to_device(gbeta)

parameters=[omega,gbeta]
parameters_global=cuda.to_device(parameters)
matrix = [0 ,0,-1, 0, 0, -1, 1, 0, 0, 1, -1, -1, 1, -1, 1, 1, -1, 1]

matrix_global=cuda.to_device(matrix)

g_global=cuda.to_device(g)
geq_global=cuda.to_device(geq)

