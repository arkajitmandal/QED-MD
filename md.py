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

def dV(R):
    # Total DOF : 2(3.M) + 1 DOF 
    # M == Molecules 
    # Each dimer has 2 atoms 
    # Each Atom has 3 dimension 
    ndof = len(R)
    M    = (R-1)//6
    N    = (R-1)//3
    dE = np.zeros((ndof))
    # Molecular Potential
    Ω  = 0.14/27.2114
    ωc = 0.07/27.2114
    χ  = 0.1 * ωc
    m1 = 1836.0 #
    m2 = 19 *  1836.0
    m  = m1 * m2 / (m1 + m2)
    Rd = np.zeros(M)
    μ  = np.zeros(M)
    for i in range(M):
        # Coordinates
        R1x, R1y, R1z =   R[6*i], R[6*i+1], R[6*i+2]
        R2x, R2y, R2z = R[6*i+3], R[6*i+4], R[6*i+5]
        # Bond-Length
        Rd[i] = ((R1x-R2x)**2 + (R1y-R2y)**2 + (R1z-R2z)**2)**0.5
        # Harmonic Potential 
        dE[2*i]     =   m * (Ω**2) *  Rd[i]
        dE[2*i + 1] = - m * (Ω**2) *  Rd[i]
        # Dipole 
        μ[i] = 
        
    # Cavity Radiation
    dE[-1] = ωc * (R[-1] + χ * (2/ωc**3.0)**0.5 


