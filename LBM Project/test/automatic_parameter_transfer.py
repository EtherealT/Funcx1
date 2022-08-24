import time
import numpy as np
from numba import cuda
from numba import njit

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


omegat = 1.0 / (3. * alpha + 0.5)
timestep = 15000
t01 = time.time()

@cuda.jit
def collision_gpu(u, v, f, feq, rho, omega, w, cx, cy, n, m, th,gbeta):

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

@njit
def collision_cpu(u=0, v=0, f=0, feq=0, rho=0, omega=0, w=0, cx=0, cy=0, n=0, m=0, th=0, gbeta=0):
    print('collision_cpu start')
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


def gpu_parameter_loading_from_cpu(**kwargs):
    """if parameter exist on CPU then copy to GPU"""
    for key in kwargs:
        try:
            """create a new parameter name with _global=  _global and assign the cpu """
            globals()[str(key) + '_global'] = cuda.to_device(kwargs[key])
            print(eval('type('+str(key)+'_global'+')'))
            print('%s_global created'%key)

        except:
            print('%s has not been initialize in CPU' % key)
            return 0

    if 1:
        return
    else:
        return


def cpu_parameter_loading_from_gpu(*args):
    """if parameter exist on GPU then retrieve"""
    for arg in args:
        try:
            print(arg)
            globals()[str(arg)]=eval(str(arg) + '_global'+'.copy_to_host')
        except:
            print('the parameter %s is not exist on GPU' % arg)
            return 0
    return 1


@cuda.jit
def test1(na, a, k):
    row, col = cuda.grid(2)

    if row < 6 and col < 6:
        na[row, col] = a
        print(na[row, col])


def test(k=19, *args):
    a, b = args


class parameters_version_dict(object):
    def __init__(self):
        self.parameter_version = {}

    def parameter_init(self, parameter_name):
        self.parameter_version[parameter_name] = 0

    def parameter_version_update(self, parameter_name):
        self.parameter_version[parameter_name] += 1

    def parameter_version_synchronize(self, parameter_name, synchronized_version):
        self.parameter_version[parameter_name] = synchronized_version

    def get_parameter_version(self, parameter_name):
        return self.parameter_version[parameter_name]


cpu_parameter_version_dict = parameters_version_dict()
gpu_parameter_version_dict = parameters_version_dict()


def parameter_version_maintain(location, *args, **kwargs):
    def gpu_parameter_version_compare_to_cpu(gpu_parameter_name):
        cpu_parameter_name = str(gpu_parameter_name).replace('_global', '')
        parameter_version_on_cpu = cpu_parameter_version_dict.get_parameter_version(cpu_parameter_name)
        print('cpu_parameter_name', cpu_parameter_name)
        """if parameter not exist in cpu dict"""
        if cpu_parameter_name in cpu_parameter_version_dict.parameter_version.keys():

            """if parameter has been init in gpu dict, then compare the parameter version between two dict"""
            if gpu_parameter_name in gpu_parameter_version_dict.parameter_version.keys():

                current_parameter_version = gpu_parameter_version_dict.get_parameter_version(str(gpu_parameter_name))
                print(gpu_parameter_name, current_parameter_version)


                print('parameter version on cpu:', parameter_version_on_cpu)
                if parameter_version_on_cpu > current_parameter_version:
                    """if parameter version in cpu dict is higher, then load the parameter to gpu, and update the version"""

                    gpu_parameter_loading_from_cpu(cpu_parameter_name=cpu_parameter_name)
                    gpu_parameter_version_dict.parameter_version_synchronize(gpu_parameter_name,
                                                                             parameter_version_on_cpu+1)

                elif parameter_version_on_cpu <= current_parameter_version:
                    """if version on gpu is higher or the same, then update the version"""
                    gpu_parameter_version_dict.parameter_version_update(gpu_parameter_name)
                else:
                    print('%s exist in gpu parameter version dict, but maintain failed' % gpu_parameter_name)
            else:
                """if parameter have not been init"""
                gpu_parameter_version_dict.parameter_init(gpu_parameter_name)
                gpu_parameter_loading_from_cpu(cpu_parameter_name=cpu_parameter_name)
                gpu_parameter_version_dict.parameter_version_synchronize(gpu_parameter_name,
                                                                         parameter_version_on_cpu + 1)
                print('%s not exist in gpu parameter dict, the version init' % cpu_parameter_name)
        else:

            if gpu_parameter_name in gpu_parameter_version_dict.parameter_version.keys():
                gpu_parameter_version_dict.parameter_version_update(gpu_parameter_name)
            else:
                gpu_parameter_version_dict.parameter_init(gpu_parameter_name)
                gpu_parameter_version_dict.parameter_version_update(gpu_parameter_name)

            print('%s created in cpu and gpu parameter version dict' % cpu_parameter_name)

    def cpu_parameter_version_compare_to_gpu(cpu_parameter_name):
        gpu_parameter_name = str(cpu_parameter_name)

        if cpu_parameter_name in cpu_parameter_version_dict.parameter_version.keys():
            if gpu_parameter_name in gpu_parameter_version_dict.parameter_version.keys():

                current_parameter_version = cpu_parameter_version_dict.get_parameter_version(str(cpu_parameter_name))
                parameter_version_on_gpu = gpu_parameter_version_dict.get_parameter_version(gpu_parameter_name)
                print(cpu_parameter_name, current_parameter_version, parameter_version_on_gpu)
                if parameter_version_on_gpu > current_parameter_version:
                    cpu_parameter_loading_from_gpu(cpu_parameter_name)
                    cpu_parameter_version_dict.parameter_version_synchronize(cpu_parameter_name,
                                                                             parameter_version_on_gpu+1)

                elif parameter_version_on_gpu <= current_parameter_version:
                    cpu_parameter_version_dict.parameter_version_update(cpu_parameter_name)
            else :
                cpu_parameter_version_dict.parameter_version_update(cpu_parameter_name)


        else:
            cpu_parameter_version_dict.parameter_init(cpu_parameter_name)
            print('%s init' % cpu_parameter_name)
            cpu_parameter_version_compare_to_gpu(cpu_parameter_name)

    if location == 'cpu':
        for key in kwargs:
            print('kwarg:', key)
            cpu_parameter_version_compare_to_gpu(key)
    elif location == 'gpu':
        for key in kwargs:
            print('kwarg:', key)
            gpu_parameter_version_compare_to_cpu(key)


def run(target_function, target_device, *args, **kwargs):
    new_func_name = target_function + '_' + target_device
    if target_device == 'cpu':
        parameter_version_maintain('cpu', **kwargs)
        print('parameter version maintain success')


        eval(new_func_name)(**kwargs)
        try:
            eval(new_func_name)(**kwargs)
            print('run %s end'%new_func_name)
        except:
            print('failed to run %s on %s' % (str(target_function), str(target_device)))
            return 0
        else:
            return 1
    elif target_device == 'gpu':
        parameter_version_maintain('gpu', **kwargs)
        TPB, BPG = args
        kwarg_value = kwargs.values()

        try:
            eval(new_func_name)[BPG,TPB](*kwarg_value)
        except:
            print('failed to run %s on %s' % (str(target_function), str(target_device)))
            return 0
        else:
            return 1
    else:
        print('please input correct target device type (cpu / gpu)')
        return 0


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
    timestep = 1500
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

    time_slice = timestep / 100

    TPB2D, BPG2D = dispatch(m, n)
    t1 = time.time()
    run('collision', 'cpu', u=u, v=v, f=f, feq=feq, rho=rho, omega=omega, w=w, cx=cx, cy=cy, n=n, m=m, th=th,
        gbeta=gbeta)
    print('gpu parameter loading')
    gpu_parameter_loading_from_cpu(u=u, v=v, f=f, feq=feq, rho=rho, omega=omega, w=w, cx=cx, cy=cy, n=n, m=m, th=th,
                                   gbeta=gbeta, g=g, geq=geq, omegat=omegat, tw=tw, matrix=matrix)

    print('collision gpu start')
    run('collision', 'gpu', TPB2D, BPG2D, u=u_global, v =v_global, f =f_global,
        feq =feq_global, rho =rho_global, omega =omega_global, w =w_global, cx =cx_global,
        cy =cy_global,
        n =n_global, m =m_global, th =th_global, gbeta =gbeta_global)
    run('collision', 'gpu', TPB2D, BPG2D, u=u_global, v=v_global, f=f_global,
        feq=feq_global, rho=rho_global, omega=omega_global, w=w_global, cx=cx_global,
        cy=cy_global,
        n=n_global, m=m_global, th=th_global, gbeta=gbeta_global)

    run('collision', 'cpu', u=u, v=v, f=f, feq=feq, rho=rho, omega=omega, w=w, cx=cx, cy=cy, n=n, m=m, th=th,
        gbeta=gbeta)
    print('cpu_parameter_version_dict',cpu_parameter_version_dict.parameter_version)
    print('gpu_parameter_version_dict:',gpu_parameter_version_dict.parameter_version)
if __name__ == '__main__':
    main()
