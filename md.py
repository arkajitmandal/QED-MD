import numpy as np
from numpy.random import normal as gran

def vv(dat):
    f1 = dat.f1
    m = dat.m 
    dt = dat.dt
    v  = dat.p/m 
    dat.x += v * dt + 0.5 * (f1/m) * (dt ** 2)   
    f2 = dat.F(dat) 
    dat.p += 0.5 * (f1 + f2) * dt 
    dat.f1 = f2
    return dat

def vvl(dat):
    ndof = dat.ndof
    β  = dat.β
    v = dat.p/dat.m
    dt = dat.dt
    λ = dat.λ #/ dat.m
    σ = (2.0 * λ/(β * dat.m )) ** 0.5
    ξ = gran(0, 1, ndof)  #np.array([0.5 * gran() for i in range(len(x))])
    θ = gran(0, 1, ndof) #np.array([gran() * 0.28867513459  for i in range(len(x))])
    c = 0.28867513459
    A = (0.5 * dt**2) * (dat.f1/dat.m - λ * v) + (σ * dt**(3.0/2.0)) * (0.5 * ξ + c * θ) 
    #---- X update -----------
    dat.x += (v * dt + A) 
    #-------------------------
    f1 = dat.f1
    f2 = dat.F(dat)
    #---- V update ----------- 
    v += ( 0.5 * dt * (f1+f2)/dat.m - dt * λ * v +  σ * (dt**0.5) * ξ - A * λ ) 
    #-------------------------
    dat.f1 = f2
    return dat

