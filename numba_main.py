def main():
    import numpy as np
    from numba import cuda

    n = 10
    m = 10

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

    TPB = 16
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
    def collision(u, v, f, feq, rho, parameter, w, cx, cy, n, m, th):

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
    def collt(u, v, g, geq, th, omegat, w, cx, cy, n, m):
        row, col = cuda.grid(2)
        if row < n + 1 and col < m + 1:
            for k in range(0, 9):
                geq[k, row, col] = th[row, col] * w[k] * (1.0 + 3.0 * (u[row, col] * cx[k] + v[row, col] * cy[k]))
                g[k, row, col] = omegat * geq[k, row, col] + (1.0 - omegat) * g[k, row, col]

    @cuda.jit
    def streaming(f, n, m, matrix):
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
    def bounceb(f, n, m):
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
    def gbound(g, tw, w, n, m):
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
    def tcalcu(g, th, n, m):
        row, col = cuda.grid(2)
        if row < n + 1 and col < m + 1:

            ssumt = 0.0
            for k in range(0, 9):
                ssumt = ssumt + g[k, row, col]
            th[row, col] = ssumt
        return

    @cuda.jit
    def rhouv(f, rho, u, v, cx, cy, n, m):
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
    timestep=1500
    for time in range(0,timestep):
        collision[BPG2D, TPB2D](u_global, v_global, f_global, feq_global, rho_global, parameters_global, w_global, cx_global, cy_global, 10, 10, th_global)
        streaming[BPG2D, TPB2D](f_global, 10, 10, matrix_global)
        bounceb[BPG2D, TPB2D](f_global, 10, 10)
        rhouv[BPG2D, TPB2D](f_global, rho_global, u_global, v_global, cx_global, cy_global, 10, 10)
        collt[BPG2D, TPB2D](u_global, v_global, g_global, geq_global, th_global, omegat, w_global, cx_global, cy_global, 10,10)
        streaming[BPG2D, TPB2D](g_global,10,10,matrix_global)
        gbound[BPG2D, TPB2D](g_global, 1, w_global, 10, 10)
        tcalcu[BPG2D, TPB2D](g_global, th_global, 10, 10)
    rho=rho_global.copy_to_host()

    return rho

if __name__ == '__main__':
    from funcx.sdk.client import FuncXClient
    import time

    fxc = FuncXClient()
    time1 = time.time()
    func_uuid = fxc.register_function(main)
    time2 = time.time()
    endpoint1 = '942cf606-bf1c-41bd-a68e-9e35eb2b588e'
    endpoint2 = 'bb05100e-3741-4f62-bde0-5793f4369499'  # Public tutorial endpoint
    res = fxc.run(function_id=func_uuid, endpoint_id=endpoint2)
    time3 = time.time()
    while fxc.get_task(res)['pending']:
        time.sleep(3)
    time4 = time.time()

    print('function uploading time:', time3 - time2)
    print('function running time: ', time4 - time3)
    print(fxc.get_result(res))
    time5 = time.time()
    print('result retrieving time:', time5 - time4)


