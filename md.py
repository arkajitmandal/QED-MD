import numpy as np
from numpy.random import normal as gran

class parameters():
    NSteps =   200 #int(2*10**6)
    dtN    =   1.0
    η      =   0.1
    ωc     =   0.10/27.2114
    Ω      =   0.14/27.2114
    M      =   10
    T = 298.0
    β = 315774/T  
    # masses
    m      =   np.zeros(M+1)
    m[::2] =    1836.0 # atom 1
    m[1::2] =   1836.0 # atom 2
    m[-1]   =   1.0    # cavity mode
    # output
    nskip  = 10

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

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
    v = dat.P/dat.m
    dt = dat.dt
    λ = dat.λ #/ dat.m
    σ = (2.0 * λ/(β * dat.m )) ** 0.5
    ξ = gran(0, 1, ndof)  #np.array([0.5 * gran() for i in range(len(x))])
    θ = gran(0, 1, ndof) #np.array([gran() * 0.28867513459  for i in range(len(x))])
    c = 0.28867513459
    A = (0.5 * dt**2) * (dat.f1/dat.m - λ * v) + (σ * dt**(3.0/2.0)) * (0.5 * ξ + c * θ) 
    #---- X update -----------
    dat.R += (v * dt + A) 
    #-------------------------
    f1 =   dat.f1
    f2 = - dat.dV(dat)
    #---- V update ----------- 
    v += ( 0.5 * dt * (f1+f2)/dat.m - dt * λ * v +  σ * (dt**0.5) * ξ - A * λ ) 
    #-------------------------
    dat.P  = v * dat.m
    dat.f1 = f2
    return dat

def dV(dat):
    R = dat.R
    # Total DOF : 2(3.M) + 1 DOF 
    # M == Molecules 
    # Each dimer has 2 atoms 
    # Each Atom has 3 dimension 
    ndof = len(R)
    M    = (R-1)//6
    N    = (R-1)//3
    dE = np.zeros((ndof))
    # Molecular Potential
    Ω  = dat.param.Ω # 0.14/27.2114
    ωc = dat.param.ωc
    χ  = dat.param.η * ωc
    m1 = dat.param.m[0] #
    m2 = dat.param.m[1]
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
        # Dipole in the Z direction
        μ0   =  0.8
        μ[i] =  μ0 * (R1z-R2z)
        # Cavity force on Molecule 
        dE[2*i]     += (ωc**2.0) * (R[-1] + χ * (2/ωc**3.0)**0.5 * np.sum(μ)) * χ * (2/ωc**3.0)**0.5 * μ0
        dE[2*i + 1] -= (ωc**2.0) * (R[-1] + χ * (2/ωc**3.0)**0.5 * np.sum(μ)) * χ * (2/ωc**3.0)**0.5 * μ0
    # Cavity Radiation
    dE[-1] = (ωc**2.0) * (R[-1] + χ * (2/ωc**3.0)**0.5 * np.sum(μ))
    # return result
    dat.dE = dE
    return dat.dE

def init(dat):
    Ω  = dat.param.Ω 
    M  = dat.param.M
    m  = dat.param.m
    R  = np.zeros((6*M+1)) 
    P  = np.zeros((6*M+1)) 
    β  = dat.param.β
    
    # Initialize in Rd 
    for i in range(M):
        σR = (1/ (β * m * Ω**2.0) ) ** 0.5
        Rd = gran(2.7023081637, σR)
        # θ  = np.random.random() * np.pi 
        # ϕ  = np.random.random() * 2.0 * np.pi 
        R[6*i+2] = +Rd/2
        R[6*i+5] = -Rd/2
    dat.R = R
    dat.P = P
    return dat
    






