import numpy as np
from numpy.random import normal as gran
import os
class parameters():
    eSteps =   50000 # equilibriation steps
    nSteps =   500000 # int(2*10**6)
    dtN    =   10.0
    η      =   4.0
    ωc     =   0.14/27.2114
    Ω      =   0.14/27.2114
    R0     =   2.7
    M      =   36
    ndof   =   6*M+2
    T = 298.0
    β = 315774/T  
    # Langevin Parameters
    λ = np.zeros((6*M+2))
    λ[:-2] = 0.05/27.2114 # not including 2 cavity modes
    # masses
    m        =   np.zeros((6*M+2))
    m[::2]   =   1836.0 # atom 1
    m[1::2]  =   1836.0 # atom 2
    m[-2]    =   1.0    # cavity mode 1
    m[-1]    =   1.0    # cavity mode 2
    # output
    nskip    = 100
    filename = "output"

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

def vv(dat):
    dt = dat.param.dtN
    f1 = dat.f1
    m = dat.param.m 
 
    v  = dat.P/m 
    dat.R += v * dt + 0.5 * (f1/m) * (dt ** 2)   
    f2 = - dV(dat)
    dat.P += 0.5 * (f1 + f2) * dt 
    dat.f1 = f2
    return dat

def vvl(dat):
    ndof = dat.param.ndof
    β  = dat.param.β
    v = dat.P/dat.param.m
    dt = dat.param.dtN
    λ =  dat.param.λ #/ dat.m
    σ = (2.0 * λ/(β * dat.param.m )) ** 0.5
    ξ = gran(0, 1, ndof)  #np.array([0.5 * gran() for i in range(len(x))])
    θ = gran(0, 1, ndof) #np.array([gran() * 0.28867513459  for i in range(len(x))])
    c = 0.28867513459
    A = (0.5 * dt**2) * (dat.f1/dat.param.m - λ * v) + (σ * dt**(3.0/2.0)) * (0.5 * ξ + c * θ) 
    #---- X update -----------
    dat.R += (v * dt + A) 
    #-------------------------
    f1 =   dat.f1
    f2 = - dV(dat)
    #---- V update ----------- 
    v += ( 0.5 * dt * (f1+f2)/dat.param.m - dt * λ * v +  σ * (dt**0.5) * ξ - A * λ ) 
    #-------------------------
    dat.P  = v * dat.param.m
    dat.f1 = f2
    return dat

def dV(dat):
    R  = dat.R
    R0 = dat.param.R0 
    # Total DOF : 2(3.M) + 1 DOF 
    # M == Molecules 
    # Each dimer has 2 atoms 
    # Each Atom has 3 dimension 
    ndof = len(R)
    M    = (ndof-2)//6
    N    = (ndof-2)//3
    dE = np.zeros((ndof))
    # Molecular Potential
    Ω  = dat.param.Ω # 0.14/27.2114
    ωc = dat.param.ωc
    χ  = dat.param.η * ωc
    m1 = dat.param.m[0] #
    m2 = dat.param.m[1]
    m  = m1 * m2 / (m1 + m2)
    Rd = np.zeros((M))
    μz  = np.zeros((M))
    μy  = np.zeros((M))



    # Dipole
    for i in range(M):
        # Dipole in the Z direction
        μ0   =  0.8
        μy[i] =  μ0 * (R[6*i+1]-R[6*i+4])
        μz[i] =  μ0 * (R[6*i+2]-R[6*i+5])

    # Cavity Radiation
    dE[-2] = (ωc**2.0) * (R[-2] + χ * (2/ωc**3.0)**0.5 * np.sum(μy))
    dE[-1] = (ωc**2.0) * (R[-1] + χ * (2/ωc**3.0)**0.5 * np.sum(μz))
    
    for i in range(M):
        # Coordinates
        R1x, R1y, R1z =   R[6*i], R[6*i+1], R[6*i+2]
        R2x, R2y, R2z = R[6*i+3], R[6*i+4], R[6*i+5]
        # Bond-Length
        Rd[i] = ((R1x-R2x)**2 + (R1y-R2y)**2 + (R1z-R2z)**2)**0.5
        # Harmonic Potential on atom 1
        dr = m * (Ω**2) * (Rd[i]-R0) * (-1/Rd[i])  
        
        dE[6*i]        =  dr * np.abs(R1x-R2x) * (((R1x-R2x) < 0)-0.5)*2 # X
        dE[6*i + 1]    =  dr * np.abs(R1y-R2y) * (((R1y-R2y) < 0)-0.5)*2 # Y
        dE[6*i + 2]    =  dr * np.abs(R1z-R2z) * (((R1z-R2z) < 0)-0.5)*2 # Z
        # Cavity Effect on atom 1 ======================================================
        # cavity mode 1
        dE[6*i + 2]    += (ωc**2.0) * (R[-2] + χ * (2/ωc**3.0)**0.5 * np.sum(μy))  \
                        * χ * (2/ωc**3.0)**0.5 * μ0
        # cavity mode 1
        dE[6*i + 2]    += (ωc**2.0) * (R[-1] + χ * (2/ωc**3.0)**0.5 * np.sum(μz))  \
                        * χ * (2/ωc**3.0)**0.5 * μ0
        #===============================================================================
        # Harmonic Potential on atom 2
        dE[6*i + 3]    = - dr * np.abs(R1x-R2x) * (((R1x-R2x) < 0)-0.5)*2 # X
        dE[6*i + 4]    = - dr * np.abs(R1y-R2y) * (((R1y-R2y) < 0)-0.5)*2 # Y
        dE[6*i + 5]    = - dr * np.abs(R1z-R2z) * (((R1z-R2z) < 0)-0.5)*2 # Z
        # Cavity Effect on atom 2 ======================================================
        # cavity mode 1
        dE[6*i + 5]    -= (ωc**2.0) * (R[-2] + χ * (2/ωc**3.0)**0.5 * np.sum(μy)) \
                        * χ * (2/ωc**3.0)**0.5 * μ0
        # cavity mode 2
        dE[6*i + 5]    -= (ωc**2.0) * (R[-1] + χ * (2/ωc**3.0)**0.5 * np.sum(μz)) \
                        * χ * (2/ωc**3.0)**0.5 * μ0
        #===============================================================================
    # return result
    dat.dE = dE
    return dE




def init(dat):
    """
    Note this is not a equilibrium distribution
    and one need to do a equilibriation run
    """
    Ω  = dat.param.Ω 
    M  = dat.param.M
    m1 = dat.param.m[0] #
    m2 = dat.param.m[1]
    m  = m1 * m2 / (m1 + m2)
    R  = np.zeros((6*M+2)) 
    P  = np.zeros((6*M+2)) 
    β  = dat.param.β
    
    # Initialize in Rd 
    for i in range(M):
        σR = (1/ (β * m * Ω**2.0) ) ** 0.5
        Rd = gran(dat.param.R0, σR) #2.7023081637
        # θ  = np.random.random() * np.pi 
        # ϕ  = np.random.random() * 2.0 * np.pi 
        R[6*i+2] = -Rd/2
        R[6*i+5] = +Rd/2
    dat.R = R
    dat.P = P
    return dat
    

def writeXYZ(dat):
    filename = dat.param.filename + ".xyz"
    M = dat.param.M
    R = dat.R[:-2] # np.reshape(dat.R[:-1],(M,3))
    atom1 = "C"
    atom2 = "O"
    txt   = f"{M*2}\n\n"
    for i in range(M):
        txt +=  f"{atom1}\t" + "\t".join(R[6*i  :6*i+3].astype(str)) + "\n"
        txt +=  f"{atom2}\t" + "\t".join(R[6*i+3:6*i+6].astype(str)) + "\n"
    fob = open(filename,"a")
    fob.write(txt)
    fob.close()

def writeR(dat):
    filename = dat.param.filename + ".txt"
    R = dat.R
    fob = open(filename,"a")
    txt = "\t".join(R.astype(str)) + "\n"
    fob.write(txt)
    fob.close()

def writeSQ(dat):
    filename = dat.param.filename + ".xyz"
    M = dat.param.M
    R = dat.R[:-2] * 1# np.reshape(dat.R[:-1],(M,3))
    atom1 = "C"
    atom2 = "O"
    txt   = f"{M*2}\n\n"
    sq = int((dat.param.M)**0.5)
    _ = 7
    for i in range(M): 

        com = 0.5 * (R[6*i:6*i+3] + (R[6*i+3:6*i+6])) 
        R1 = R[6*i  :6*i+3] - com
        R2 = R[6*i+3:6*i+6] - com

        R1[0] += (i%sq) * _
        R1[2] += int(i/sq) * _
        R2[0] += (i%sq) * _
        R2[2] += int(i/sq) * _

        txt +=  f"{atom1}\t" + "\t".join(R1.astype(str)) + "\n"
        txt +=  f"{atom2}\t" + "\t".join(R2.astype(str)) + "\n"
    fob = open(filename,"a")
    fob.write(txt)
    fob.close()

def writeθ(dat):
    """
    M columns for θ of M molecules
    and last column show the standard 
    deviation
    """
    filename = dat.param.filename + ".theta"
    R = dat.R
    θ = np.zeros((dat.param.M+1))
    for i in range(dat.param.M):
        # Coordinates
        R1x, R1y, R1z =   R[6*i], R[6*i+1], R[6*i+2]
        R2x, R2y, R2z = R[6*i+3], R[6*i+4], R[6*i+5]
        # Bond-Length
        Rd = ((R1x-R2x)**2 + (R1y-R2y)**2 + (R1z-R2z)**2)**0.5
        Ryz = ((R1z-R2z)**2 + (R1y-R2y)**2)**0.5
        θ[i] = np.arccos(Ryz/Rd) 
    θ[-1] = np.std(θ[:-1])
    fob = open(filename,"a")
    txt = "\t".join(θ.astype(str)) + "\n"
    fob.write(txt)
    fob.close()

def run(param):

    os.system(f"rm -rf {param.filename}*")

    # Equilibriation
    datEql =  Bunch(param = param)
    datEql.param.λ = 0.005/27.2114 #1
    datEql.param.η      =   0.0 # no coupling for equilibrium
    datEql =  init(datEql)
    datEql.dV =  dV
    # Init force
    datEql.f1 = -dV(datEql)
    for t in range(datEql.param.eSteps):
        datEql = vvl(datEql)
        if (t%datEql.param.nskip==0):
            #writeXYZ(datEql)
            #writeSQ(datEql)
            writeθ(datEql)

    # Main NVT
    datNVT =  Bunch(param = param)
    datNVT.param.λ *= 0.0#1
    datNVT.param.β *= 10.0
    datNVT.R, datNVT.P = datEql.R * 1.0, datEql.P * 1.0
    
    datNVT.dV =  dV
    
    # Init force
    datNVT.f1 = -dV(datNVT)
    for t in range(datNVT.param.nSteps):
        datNVT = vvl(datNVT)
        if (t%datNVT.param.nskip==0):
            #writeXYZ(datEql)
            writeSQ(datNVT)
            writeθ(datNVT)




param = parameters()
run(param)