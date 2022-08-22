import numpy as np
import numba
from numba import njit
import time

@njit
def collision_cpu(u,v,f,feq,rho,omega,w,cx,cy,n,m,th,gbeta):
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

@njit
def collt_cpu(u,v,g,geq,th,omegat,w,cx,cy,n,m):
    for i in range(0,n+1):
        for j in range(0,m+1):
            for k in range(0,9):
                geq[k,i,j]=th[i,j]*w[k]*(1.0+3.0*(u[i,j]*cx[k]+v[i,j]*cy[k]))
                g[k,i,j]=omegat*geq[k,i,j]+(1.0-omegat)*g[k,i,j]
    return

@njit
def streaming_cpu(f,n,m):
    for j in range (0,m+1):
        for i in range (n,0,-1):
            f[1,i,j]=f[1,i-1,j]
        for i in range (0,n):
            f[3,i,j]=f[3,i+1,j]
    for j in range (m,0,-1):
        for i in range(0,n+1):
            f[2,i,j]=f[2,i,j-1]
        for i in range (n,0,-1):
            f[5,i,j]=f[5,i-1,j-1]
        for i in range(0,n):
            f[6,i,j]=f[6,i+1,j-1]
    for j in range (0,m):
        for i in range (0,n+1):
            f [4,i,j]=f[4,i,j+1]
        for i in range(0,n):
            f[7,i,j]=f[7,i+1,j+1]
        for i in range(n,0,-1):
            f[8,i,j]=f[8,i-1,j+1]
    return

@njit
def bounceb_cpu(f,n,m):
    for j in range(0, m+1):
        '''
        west boundary
        '''
        f[1, 0, j] = f[3, 0, j]
        f[5, 0, j] = f[7, 0, j]
        f[8, 0, j] = f[6, 0, j]
        '''
        east boundary
        '''
        f[3, n, j] = f[1, n, j]
        f[7, n, j] = f[5, n, j]
        f[6, n, j] = f[8, n, j]
    for i in range(0,n+1):
        '''
             south boundary
        '''
        f[2 ,i, 0] = f[4, i, 0]
        f[5, i, 0] = f[7, i, 0]
        f[6, i, 0] = f[8, i, 0]
        '''
            north boundary
        '''
        f[4, i, m] = f[2, i, m]
        f[8, i, m] = f[6, i, m]
        f[7, i, m] = f[5, i, m]
    return

@njit
def gbound_cpu(g,tw,w,n,m):
    for j in range(0,m+1):
        g[1, 0, j] = tw * (w[1] + w[3]) - g[3, 0, j]
        g[5, 0, j] = tw * (w[5] + w[7]) - g[7, 0, j]
        g[8, 0, j] = tw * (w[8] + w[6]) - g[6, 0, j]
    for j in range(0,m+1):
        g[6, n, j] = -g[8, n, j]
        g[3, n, j] = -g[1, n, j]
        g[7, n, j] = -g[5, n, j]
    for i in range(0,n+1):
        g[8, i, m] = g[8, i, m-1]
        g[7, i, m] = g[7, i, m - 1]
        g[6, i, m] = g[6, i, m - 1]
        g[5, i, m] = g[5, i, m - 1]
        g[4, i, m] = g[4, i, m - 1]
        g[3, i, m] = g[3, i, m - 1]
        g[2, i, m] = g[2, i, m - 1]
        g[1, i, m] = g[1, i, m - 1]
        g[0, i, m] = g[0, i, m - 1]
    for i in range (0,n+1):
        g[1, i, 0] = g[1, i, 1]
        g[2, i, 0] = g[2, i, 1]
        g[3, i, 0] = g[3, i, 1]
        g[4, i, 0] = g[4, i, 1]
        g[5, i, 0] = g[5, i, 1]
        g[6, i, 0] = g[6, i, 1]
        g[7, i, 0] = g[7, i, 1]
        g[8, i, 0] = g[8, i, 1]
        g[0, i, 0] = g[0, i, 1]
    return

@njit
def tcalcu_cpu(g,th,n,m):
    for j in range(0,m+1):
        for i in range(0,n+1):
            ssumt=0.0
            for k in range (0,9):
                ssumt=ssumt+g[k,i,j]
            th[i,j]=ssumt
    return

@njit
def rhouv_cpu(f,rho,u,v,cx,cy,n,m):
    for j in range(0,m+1):
        for i in range(0,n+1):
            ssum=0.0
            for k in range (0,9):
                ssum=ssum+f[k,i,j]
            rho[i,j]=ssum
    for i in range(0, n + 1):
        for j in range(0, m + 1):
            usum=0.0
            vsum=0.0
            for k in range(0,9):
                usum=usum+f[k,i,j]*cx[k]
                vsum=vsum+f[k,i,j]*cy[k]
            u[i,j]=usum/rho[i,j]
            v[i,j]=vsum/rho[i,j]
    return

@njit
def result_cpu(u,v,rho,th,uo,n,m,ra):
    strf=np.full((n+1,m+1),0,np.float)
    for i in range(0,n+1):
        rhoav=0.5*(rho[i-1,0]+rho[i,0])
        if (i!=0):
            strf[i,0]=strf[i-1,0]-rhoav*0.5*(v[i-1,0]+v[i,0])
    return


if __name__ == '__main__':

    t0=time.time()
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
    timestep=15000
    t01 = time.time()
    collision_time=0
    streaming_time=0
    bounceb_time=0
    rhouv_time=0
    collt_time=0
    g_streaming_time=0
    gbound_time=0
    tcalcu_time=0

    for mytime in range(0,timestep):
        t1=time.time()
        collision_cpu(u,v,f,feq,rho,omega,w,cx,cy,n,m,th,gbeta)
        t2 = time.time()
        streaming_cpu(f, n,m)
        t3 = time.time()
        bounceb_cpu(f, n,m)
        t4 = time.time()
        rhouv_cpu(f,rho,u,v,cx,cy,n,m)
        t5 = time.time()
        collt_cpu(u, v, g, geq, th, omegat, w, cx, cy, n,m)
        t6 = time.time()
        streaming_cpu(g,n,m)
        t7 = time.time()
        gbound_cpu(g,tw,w,n,m)
        t8 = time.time()
        tcalcu_cpu(g,th,n,m)
        t9 = time.time()
        collision_time = collision_time + t2 - t1
        streaming_time = streaming_time + t3 - t2
        bounceb_time = bounceb_time + t4 - t3
        rhouv_time = rhouv_time + t5 - t4
        collt_time = collt_time + t6 - t5
        g_streaming_time = g_streaming_time + t7 - t6
        gbound_time = gbound_time + t8 - t7
        tcalcu_time = tcalcu_time + t9 - t8
    print(rho)
    t02=time.time()
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