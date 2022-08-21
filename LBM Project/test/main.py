
from CPU2 import *
from numba import cuda

import time



@cuda.jit
def collision_gpu(u, v, f, feq, rho, parameter, w, cx, cy, n, m, th):
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



def run(target_function, target_device,*args, **kwargs):
    new_func_name = target_function + '_' + target_device
    if target_device == 'cpu':

        try:
            eval(new_func_name)(**kwargs)
        except:
            print('failed to run %s on %s' % (str(target_function), str(target_device)))
            return 0
        else:
            return 1
    elif target_device == 'gpu':


        eval(new_func_name)[10,5](*args)
        try:
            eval(new_func_name)[10,5](*args)
        except:
            print('failed to run %s on %s' % (str(target_function), str(target_device)))
            return 0
        else:
            return 1
    else:
        print('please input correct target device type (cpu / gpu)')
        return 0


def collision_cpu(u=0,v=0,f=0,feq=0,rho=0,omega=0,w=0,cx=0,cy=0,n=0,m=0,th=0,gbeta=0):
    tref=0.5
    for i in range(0,n+1):
        for j in range (0,m+1):
            t1=u[i,j]*u[i,j]+v[i,j]*v[i,j]
            for k in range (0,9):
                t2=u[i,j]*cx[k]+v[i,j]*cy[k]

                force=3.*w[k]*gbeta*(th[i,j]-tref)*cy[k]*rho[i,j]
                if(i==0 or i==n):
                    force =0.0
                if(j==0 or j==m):
                    force=0.0
                feq[k,i,j]=rho[i,j]*w[k]*(1+3*t2+4.50*t2*t2-1.5*t1)
                f[k,i,j]=omega*feq[k,i,j]+(1.-omega)*f[k,i,j]+force

    return


def gpu_parameter_loading_from_cpu(**kwargs):
    """if parameter exist on CPU then copy to GPU"""
    for key in kwargs:
        try:
            """create a new parameter name with _global and assign the cpu """
            globals()[str(key) + '_global'] = cuda.to_device(key)
        except:
            print('%s has not been initialize in CPU' % key)
            return 0

    return 1



def cpu_parameter_loading_from_gpu(**kwargs):
    """if parameter exist on GPU then retrieve"""
    for key in kwargs:
        func_name_on_gpu = str(key) + '_global'
        """tell if exist on GPU"""
        try:
            parameter_type = type(func_name_on_gpu)
            """if on GPU"""
            if 'cuda' in str(parameter_type):
                kwargs[key] = func_name_on_gpu.copy_to_host
                """if not on GPU"""
            else:
                print('%s is not on GPU, the original cpu version of %s is used' % (key, key))

        except:
            print('the parameter is not exist on both CPU and GPU')
            return 0
    return 1


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
    collision_time = 0
    streaming_time = 0
    bounceb_time = 0
    rhouv_time = 0
    collt_time = 0
    g_streaming_time = 0
    gbound_time = 0
    tcalcu_time = 0
    print('f1',f)
    gpu_parameter_loading_from_cpu(u=u,v=v,f=f,feq=feq,rho=rho,w=w,cx=cx,cy=cy,n=n,m=m,th=th,parameters=parameters)


    run ('collision','gpu',u_global,v_global,f_global,feq_global,rho_global,parameters_global,w_global,cx_global,cy_global,n_global,m_global,th_global)
    f=f_global.copy_to_host
    print('f3',f)
    cpu_parameter_loading_from_gpu(u=u_global,v=v_global,f=f_global,feq=feq_global,rho=rho_global,omega=omega_global,w=w_global,cx=cx_global,cy=cy_global,n=n_global,m=m_global,th=th_global,gbeta=gbeta_global)

    print(f)
def test(*args):
    a, b=args
    print(a,b)
if __name__ == '__main__':
    main()