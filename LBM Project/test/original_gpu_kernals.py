import numpy as np
from numba import cuda
import time


n = 10
m = 10

rhoo = 6.00

feq = np.full((9, n + 1, m + 1), 0, float)
fin = np.full((9, n + 1, m + 1), 0, float)
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

omegat = 1.0 / (3. * alpha + 0.5)
mstep = 15


SM = 10


def dispatch(m, n):
    threadsperblock = (16, 16)
    blockspergrid_x = np.math.ceil(m / threadsperblock[0])
    blockspergrid_y = np.math.ceil(n / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    return threadsperblock, blockspergrid


TPB2D, BPG2D = dispatch(n, m)
u_global = cuda.to_device(u)
v_global = cuda.to_device(v)
f_global = cuda.to_device(fin)
feq_global = cuda.to_device(feq)
rho_global = cuda.to_device(rho)

omega_global = cuda.to_device(omega)
w_global = cuda.to_device(w)
cx_global = cuda.to_device(cx)
cy_global = cuda.to_device(cy)
n_global = cuda.to_device(n)
m_global = cuda.to_device(m)
th_global = cuda.to_device(th)
gbeta_global = cuda.to_device(gbeta)

parameters = [omega, gbeta]
parameters_global = cuda.to_device(parameters)
matrix = [0, 0, -1, 0, 0, -1, 1, 0, 0, 1, -1, -1, 1, -1, 1, 1, -1, 1]

matrix_global = cuda.to_device(matrix)

g_global = cuda.to_device(g)
geq_global = cuda.to_device(geq)


@cuda.jit
def collision_gpu(u=0, v=0, f=0, feq=0, rho=0, parameter=0, w=0, cx=0, cy=0, n=0, m=0, th=0):
    omega = parameter[0]
    gbeta = parameter[1]
    tref = 0.5
    row, col = cuda.grid(2)
    if row < n + 1 and col < m + 1:
        t1 = u[row, col] * u[row, col] + v[row, col] * v[row, col]
        for k in range(9):
            t2 = u[row, col] * cx[k] + v[row, col] * cy[k]
            force = 3. * w[k] * gbeta * (th[row, col] - tref) * cy[k] * rho[row, col]
            if (row == 0 or row == n):
                force = 0.0
            if (col == 0 or col == m):
                force = 0.0
            feq[k, row, col] = rho[row, col] * w[k] * (1.0 + 3.0 * t2 + 4.50 * t2 * t2 - 1.50 * t1)
            f[k, row, col] = omega * feq[k, row, col] + (1. - omega) * f[k, row, col] + force


@cuda.jit
def collt_gpu(u, v, g, geq, th, omegat, w, cx, cy, n, m):
    row, col = cuda.grid(2)
    if row < n + 1 and col < m + 1:
        for k in range(0, 9):
            geq[k, row, col] = th[row, col] * w[k] * (1.0 + 3.0 * (u[row, col] * cx[k] + v[row, col] * cy[k]))
            g[k, row, col] = omegat * geq[k, row, col] + (1.0 - omegat) * g[k, row, col]


@cuda.jit
def streaming_gpu(f, n, m, matrix):
    row, col = cuda.grid(2)
    if row < n + 1 and col < m + 1:
        for k in range(1, 9):
            i = row + matrix[2 * k]
            j = col + matrix[2 * k + 1]
            if i == n + 1:
                i = row
                j = col
            if i == -1:
                i = row
                j = col
            if j == m + 1:
                j = col
                i = row
            if j == -1:
                j = col
                i = row
            f[k, row, col] = f[k, i, j]


@cuda.jit
def bounceb_gpu(f, n, m):
    row, col = cuda.grid(2)
    if row < n + 1 and col < m + 1:
        '''
            west boundary
        '''
        if row == 0:
            f[1, row, col] = f[3, row, col]
            f[5, row, col] = f[7, row, col]
            f[8, row, col] = f[6, row, col]
        '''
            east boundary
        '''
        if row == n:
            f[3, row, col] = f[1, row, col]
            f[7, row, col] = f[5, row, col]
            f[6, row, col] = f[8, row, col]

        if col == 0:
            '''
             south boundary
            '''
            f[2, row, col] = f[4, row, col]
            f[5, row, col] = f[7, row, col]
            f[6, row, col] = f[8, row, col]
        if col == m:
            '''
             north boundary
            '''
            f[4, row, col] = f[2, row, col]
            f[8, row, col] = f[6, row, col]
            f[7, row, col] = f[5, row, col]
    return


@cuda.jit
def gbound_gpu(g, tw, w, n, m):
    row, col = cuda.grid(2)
    if row < n + 1 and col < m + 1:
        if row == 0:
            g[1, row, col] = tw * (w[1] + w[3]) - g[3, row, col]
            g[5, row, col] = tw * (w[5] + w[7]) - g[7, row, col]
            g[8, row, col] = tw * (w[8] + w[6]) - g[6, row, col]
        if row == n:
            g[6, row, col] = -g[8, row, col]
            g[3, row, col] = -g[1, row, col]
            g[7, row, col] = -g[5, row, col]
        if col == m:
            for k in range(0, 9):
                g[k, row, col] = g[k, row, col - 1]
        if col == 0:

            for k in range(0, 9):
                g[k, row, col] = g[k, row, col + 1]

    return


@cuda.jit
def tcalcu_gpu(g, th, n, m):
    row, col = cuda.grid(2)
    if row < n + 1 and col < m + 1:

        ssumt = 0.0
        for k in range(0, 9):
            ssumt = ssumt + g[k, row, col]
        th[row, col] = ssumt
    return


@cuda.jit
def rhouv_gpu(f, rho, u, v, cx, cy, n, m):
    row, col = cuda.grid(2)
    if row < n + 1 and col < m + 1:
        ssum = 0.0
        for k in range(0, 9):
            ssum = ssum + f[k, row, col]
        rho[row, col] = ssum
        usum = 0.0
        vsum = 0.0
        for k in range(0, 9):
            usum = usum + f[k, row, col] * cx[k]
            vsum = vsum + f[k, row, col] * cy[k]
        u[row, col] = usum / rho[row, col]
        v[row, col] = vsum / rho[row, col]
    return

if __name__ == '__main__':

    timestep = 1500
    t1 = time.time()

    for time in range(0, timestep):
        collision_gpu[BPG2D, TPB2D](u_global, v_global, f_global, feq_global, rho_global, parameters_global, w_global,
                                cx_global, cy_global, 10, 10, th_global)
        streaming_gpu[BPG2D, TPB2D](f_global, 10, 10, matrix_global)
        bounceb_gpu[BPG2D, TPB2D](f_global, 10, 10)
        rhouv_gpu[BPG2D, TPB2D](f_global, rho_global, u_global, v_global, cx_global, cy_global, 10, 10)
        collt_gpu[BPG2D, TPB2D](u_global, v_global, g_global, geq_global, th_global, omegat, w_global, cx_global, cy_global,
                            10,
                            10)
        streaming_gpu[BPG2D, TPB2D](g_global, 10, 10, matrix_global)
        gbound_gpu[BPG2D, TPB2D](g_global, 1, w_global, 10, 10)
        tcalcu_gpu[BPG2D, TPB2D](g_global, th_global, 10, 10)
    f = f_global.copy_to_host

    t2 = time.time()
    print('t2:', t2)
    print('t1', t1)
    print('time:', t2 - t1)
