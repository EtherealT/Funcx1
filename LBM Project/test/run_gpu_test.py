from run_cpu_test import *
from numba import cuda
from numba import njit
import sys
import time


@njit
def collision_cpu(u=0, v=0, f=0, feq=0, rho=0, omega=0, w=0, cx=0, cy=0, n=0, m=0, th=0, gbeta=0):
    tref = 0.5
    for i in range(0, n + 1):
        for j in range(0, m + 1):
            t1 = u[i, j] * u[i, j] + v[i, j] * v[i, j]
            for k in range(0, 9):
                t2 = u[i, j] * cx[k] + v[i, j] * cy[k]

                force = 3. * w[k] * gbeta * (th[i, j] - tref) * cy[k] * rho[i, j]
                if (i == 0 or i == n):
                    force = 0.0
                if (j == 0 or j == m):
                    force = 0.0
                feq[k, i, j] = rho[i, j] * w[k] * (1 + 3 * t2 + 4.50 * t2 * t2 - 1.5 * t1)
                f[k, i, j] = omega * feq[k, i, j] + (1. - omega) * f[k, i, j] + force

    return


@cuda.jit
def collision_gpu(u, v, f, feq, rho, omega, w, cx, cy, n, m, th, gbeta):
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


def run(target_function, target_device, *args, **kwargs):
    new_func_name = target_function + '_' + target_device
    if target_device == 'cpu':
        try:
            eval(new_func_name)(**kwargs)
            # print('run %s on %s successfully'%(target_function,target_device))
        except:
            print('failed to run %s on %s' % (str(target_function), str(target_device)))
            return 0
        else:
            return 1
    elif target_device == 'gpu':
        for key in kwargs:
            if key == 'TPB':
                TPB = kwargs[key]
            elif key == 'BPG':
                BPG = kwargs[key]
            else:
                print('key %s cannot be recognized' % key)
        # print('TPB,BPG:', TPB, BPG)
        eval(new_func_name)[BPG, TPB](*args)
        try:
            eval(new_func_name)[BPG, TPB](*args)
            # print('run %s on %s successfully' % (target_function, target_device))
        except:
            print('failed to run %s on %s' % (str(target_function), str(target_device)))
            return 0
        else:
            return 1
    else:
        print('please input correct target device type (cpu / gpu)')
        return 0


def gpu_parameter_loading_from_cpu(**kwargs):
    """if parameter exist on CPU then copy to GPU"""
    for key in kwargs:

        try:
            if 'float' in str(type(kwargs[key])) or 'int' in str(type(kwargs[key])):
                """float and int type cannot load to GPU, but they could be directly used by GPU"""
                globals()[str(key) + '_global'] = kwargs[key]


            else:
                """create a new parameter name with _global and assign the cpu """
                globals()[str(key) + '_global'] = cuda.to_device(kwargs[key])
            # print('%s load to GPU successfully'%key)

        except:
            print('%s has not been initialize in CPU' % key)
            return 0

    return 1


def cpu_parameter_loading_from_gpu(**kwargs):
    """if parameter exist on GPU then retrieve"""
    for key in kwargs:
        func_name_on_gpu = eval(str(key) + '_global')
        """tell if exist on GPU"""
        try:
            parameter_type = type(func_name_on_gpu)
            """if on GPU"""
            if 'cuda' in str(parameter_type):
                kwargs[key] = func_name_on_gpu.copy_to_host
                """if not on GPU"""
                print('%s load to cpu successfully' % key)
            else:
                print('%s is not on GPU, the original cpu version of %s is used' % (key, key))
        except:
            print('the parameter %s is not exist on both CPU and GPU' % key)
            return 0
    return 1

def progress_bar():
    for i in range(1, 101):
        print("\r", end="")
        print("program progress: {}%: ".format(i), "▋" * (i // 2), end="")
        sys.stdout.flush()
        time.sleep(0.05)

def main():
    print('the program start:')
    t0 = time.time()
    n = 1000
    m = 1000

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
    omegat = 1.0 / (3. * alpha + 0.5)
    timestep = 15000
    matrix = [0, 0, -1, 0, 0, -1, 1, 0, 0, 1, -1, -1, 1, -1, 1, 1, -1, 1]

    def dispatch(m, n):
        threadsperblock = (16, 16)
        blockspergrid_x = np.math.ceil(m / threadsperblock[0])
        blockspergrid_y = np.math.ceil(n / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        return threadsperblock, blockspergrid

    t01 = time.time()
    parameter_load_to_gpu_time = 0
    collision_time = 0
    streaming_time = 0
    bounceb_time = 0
    rhouv_time = 0
    collt_time = 0
    g_streaming_time = 0
    gbound_time = 0
    tcalcu_time = 0

    time_slice=timestep/100

    gpu_parameter_loading_from_cpu(u=u, v=v, f=f, feq=feq, rho=rho, omega=omega, w=w, cx=cx, cy=cy, n=n, m=m, th=th,
                                   gbeta=gbeta, g=g, geq=geq, omegat=omegat, tw=tw, matrix=matrix)

    for mytime in range(0, timestep):
        if mytime%time_slice==0:
            print("\r", end="")
            print("program progress: {}%: ".format(mytime/time_slice), end="",flush=True)


        TPB2D, BPG2D = dispatch(m, n)
        t1 = time.time()
        t2 = time.time()
        run('collision', 'gpu', u_global, v_global, f_global, feq_global, rho_global, omega_global, w_global, cx_global,
            cy_global,
            n_global, m_global, th_global, gbeta_global, TPB=TPB2D, BPG=BPG2D)
        t21 = time.time()
        run('streaming', 'gpu', f_global, n_global, m_global, matrix_global, TPB=TPB2D, BPG=BPG2D)
        t3 = time.time()
        run('bounceb', 'gpu', f_global, n_global, m_global, TPB=TPB2D, BPG=BPG2D)
        t4 = time.time()
        run('rhouv', 'gpu', f_global, rho_global, u_global, v_global, cx_global, cy_global, n_global, m_global,
            TPB=TPB2D, BPG=BPG2D)
        t5 = time.time()
        run('collt', 'gpu', u_global, v_global, g_global, geq_global, th_global, omegat_global, w_global, cx_global,
            cy_global, n_global, m_global, TPB=TPB2D, BPG=BPG2D)
        t6 = time.time()
        run('streaming', 'gpu', g_global, n_global, m_global, matrix_global, TPB=TPB2D, BPG=BPG2D)
        t7 = time.time()
        run('gbound', 'gpu', g_global, tw_global, w_global, n_global, m_global, TPB=TPB2D, BPG=BPG2D)
        t8 = time.time()
        run('tcalcu', 'gpu', g_global, th_global, n_global, m_global, TPB=TPB2D, BPG=BPG2D)
        t9 = time.time()

        parameter_load_to_gpu_time = parameter_load_to_gpu_time + t2 - t1
        collision_time = collision_time + t21 - t2
        streaming_time = streaming_time + t3 - t2
        bounceb_time = bounceb_time + t4 - t3
        rhouv_time = rhouv_time + t5 - t4
        collt_time = collt_time + t6 - t5
        g_streaming_time = g_streaming_time + t7 - t6
        gbound_time = gbound_time + t8 - t7
        tcalcu_time = tcalcu_time + t9 - t8
    rho = rho_global.copy_to_host()
    print('rho', rho)
    t02 = time.time()
    print('data generation:', t01 - t0)
    print('collision:', collision_time)
    print('streaming', streaming_time)
    print('bounceb', bounceb_time)
    print('rhouv:', rhouv_time)
    print('collt', collt_time)
    print('g_streaming', g_streaming_time)
    print('g_bound', gbound_time)
    print('tcalcu', tcalcu_time)
    print('total time:', t02 - t0)

    #
    # time1=time.time()
    # gpu_parameter_loading_from_cpu(u=u,v=v,f=f,feq=feq,rho=rho,omega=omega,w=w,cx=cx,cy=cy,n=n,m=m,th=th,gbeta=gbeta)
    # time2=time.time()
    # run ('collision','gpu',u_global,v_global,f_global,feq_global,rho_global,omega_global,w_global,cx_global,cy_global,n_global,m_global,th_global,gbeta_global,TPB=2,BPG=3)
    # time3=time.time()
    # run('')
    # cpu_parameter_loading_from_gpu(u=u,v=v,f=f,feq=feq,rho=rho,omega=omega,w=w,cx=cx,cy=cy,n=n,m=m,th=th,gbeta=gbeta)
    # time4 = time.time()


def test(*args):
    a, b = args
    print(a, b)


if __name__ == '__main__':
    main()
