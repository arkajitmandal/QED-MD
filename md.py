import numpy as np
from numpy.random import normal as gran

def vv(x, p, param):
    f1 = param.force(x, param) 
    m = param.m 
    dt = param.dt
    v  = p/m 
    x += v * dt + 0.5 * (f1/m) * (dt ** 2)   
    f2 = param.force(x, param) 
    p += 0.5 * (f1 + f2) * dt 
    return x, p, f2

def vvl(x, p, param, f1 = "DO" ):
    if f1=="DO":
        f1 = param.force(x, param)
    ndof = param.ndof
    β  = param.β
    v = p/param.m
    dt = param.dt
    λ = param.λ #/ param.m
    σ = (2.0 * λ/(β * param.m )) ** 0.5
    ξ = gran(0, 1, ndof)  #np.array([0.5 * gran() for i in range(len(x))])
    θ = gran(0, 1, ndof) #np.array([gran() * 0.28867513459  for i in range(len(x))])
    c = 0.28867513459
    A = (0.5 * dt**2) * (f1/param.m - λ * v) + (σ * dt**(3.0/2.0)) * (0.5 * ξ + c * θ) 
    #---- X update -----------
    x += (v * dt + A) 
    #-------------------------
    f2 = param.force(x, param)
    #---- V update ----------- 
    v += ( 0.5 * dt * (f1+f2)/param.m - dt * λ * v +  σ * (dt**0.5) * ξ - A * λ ) 
    #-------------------------
    return x, v * param.m, f2