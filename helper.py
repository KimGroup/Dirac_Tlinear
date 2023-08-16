from __future__ import division

import numpy as np
from numpy.random import random
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz

import time
from datetime import datetime
from os import path
import h5py

import matplotlib.pyplot as plt

from matplotlib.patches import RegularPolygon


def GetNNTriangular(i, j, N):
    if i%2 == 0:
        NN_list = [[i, (j+1)%N], [(i-1)%N, j], [(i-1)%N, (j-1)%N], [i, (j-1)%N], [(i+1)%N, (j-1)%N], [(i+1)%N, j]]
    else:
        NN_list = [[i, (j+1)%N], [(i-1)%N, (j+1)%N], [(i-1)%N, j], [i, (j-1)%N], [(i+1)%N, j], [(i+1)%N, (j+1)%N]]
    return NN_list

def GetN3Triangular(i, j, N):
    if i%2 == 0:
        N3_list = [[(i-1)%N, (j+1)%N], [(i-2)%N, j], [(i-1)%N, (j-2)%N], [(i+1)%N, (j-2)%N], [(i+2)%N, j], [(i+1)%N, (j+1)%N]]
    else:
        N3_list = [[(i-1)%N, (j+2)%N], [(i-2)%N, j], [(i-1)%N, (j-1)%N], [(i+1)%N, (j-1)%N], [(i+2)%N, j], [(i+1)%N, (j+2)%N]]
    return N3_list





class IsingNN():
    def __init__(self, N, temp, config=[]):
        self.N = N
        self.beta = 1./temp
        
        if len(config) == 0:
            self.config = np.random.choice([1, -1], size=(N, N))
        else:
            self.config = config
        self.M = np.sum(self.config.flatten())
        
        
    def Clustering(self, i, j, s, flip_stack):
        N = self.N
        config = self.config
        beta = self.beta
        
        NN_list = GetNNTriangular(i, j, N)
        
        for [NNi, NNj] in NN_list:
            if (config[NNi, NNj] == s) and (random() < (1.-np.exp(-2*beta))):
                config[NNi, NNj] *= -1
                flip_stack.append([NNi, NNj])
        return flip_stack
        
    def MCSteps(self, Nstep):
        
        N = self.N
        config = self.config
        beta = self.beta
        
        
        for i in range(Nstep):
            a = np.random.randint(0, N)
            b = np.random.randint(0, N)
            
            s = config[a, b]
            flip_stack = [[a,b]]
            config[a,b] *= -1
            
            while len(flip_stack) > 0:
                [j, k] = flip_stack.pop(0)

                flip_stack = self.Clustering(j, k, s, flip_stack)
    
    def GetConfigs(self, Nwarmup, Ncycle, Lcycle):

        self.MCSteps(Nwarmup)
        
        configs = np.zeros((Ncycle, N, N), dtype=int)

        for i in range(Ncycle):
            self.MCSteps(Lcycle)
            configs[i] = np.array(self.config)
            
        return configs
    
    def GetM(self):
        return np.sum(self.config.flatten())
    
    def GetE(self):
        N = self.N
        E = 0.
        
        for i in range(N):
            for j in range(N):
                NN_list = GetNNTriangular(i, j, N)
                dE = 0
                for k in range(3):
                    [NNi, NNj] = NN_list[k]
                    dE += self.config[NNi, NNj]
                E += -self.config[i, j] * dE
        return E/N**2
        
    def GetS_slow(self, omega, eps):
        N = self.N
        config = self.config
        beta = self.beta
        
        S = 0.
        
        for i in range(N):
            for j in range(N):
                NN_list = GetNNTriangular(i, j, N)
                dE = 0
                for k in range(len(NN_list)):
                    [NNi, NNj] = NN_list[k]
                    dE += self.config[NNi, NNj]
                Em = 2*self.config[i, j] * dE
                
                if Em == 0:
                    BF = beta
                else:
                    BF = (1-np.exp(-beta*Em))/Em
                    
                S += BF*(eps/((omega-Em)**2 + eps**2))
        return S/N**2
    
    def GetS(self, omega, eps):
        N = self.N
        config = self.config
        beta = self.beta
        
        Em = np.zeros((N**2), dtype=float)
        
        l = 0
        for i in range(N):
            for j in range(N):
                NN_list = GetNNTriangular(i, j, N)
                dE = 0
                for k in range(len(NN_list)):
                    [NNi, NNj] = NN_list[k]
                    dE += self.config[NNi, NNj]
                Em[l] = 2*self.config[i, j] * dE
                l += 1
    
        return np.mean(np.divide((1-np.exp(-beta*Em)), Em, out=np.full((N**2), beta), where=Em!=0) * (eps/((omega-Em)**2 + eps**2)))
    
    def Measure(self, Nwarmup, Ncycle, Lcycle, omega=0., eps=0.05):
        
        self.MCSteps(Nwarmup)
        
        M_list = np.zeros((Ncycle), dtype=float)
        E_list = np.zeros((Ncycle), dtype=float)
        S_list = np.zeros((Ncycle), dtype=float)
        
        for i in range(Ncycle):
            self.MCSteps(Lcycle)
            
#             M_list[i] = self.GetM()
            E_list[i] = self.GetE()
            S_list[i] = self.GetS(omega, eps)
            
        return M_list, E_list, S_list

def GetNNTriOpen(i, j, Nr, Nc):
    if i%2 == 0:
        NN_list = np.array([[i, j+1], [i-1, j], [i-1, j-1], [i, j-1], [i+1, j-1], [i+1, j]])
    else:
        NN_list = np.array([[i, j+1], [i-1, j+1], [i-1, j], [i, j-1], [i+1, j], [i+1, j+1]])

    delete = []

    for di in range(6):
        if (NN_list[di, 0] < 0) or (NN_list[di, 0] >= Nr) or (NN_list[di, 1] < 0) or (NN_list[di, 1] >= Nc):
            delete.append(di)


    return np.delete(NN_list, delete, axis=0)


def GetTriDistances(Nr, Nc):
    N = Nr*Nc
    dist = np.zeros((N, N), dtype=float)

    for i1 in range(Nr):
        for j1 in range(Nc):
            for i2 in range(Nr):
                for j2 in range(Nc):
                    ij1 = i1*Nc+j1
                    ij2 = i2*Nc+j2
                    
                    dr = (i2-i1)*np.sqrt(3)/2
                    
                    if (i2-i1)%2 == 0:
                        dc = (j2-j1)
                    elif i1%2 == 0:
                        dc = (j2-j1) + 1/2
                    else:
                        dc = (j2-j1) - 1/2

                    dist[ij1, ij2] = np.sqrt(dr**2+dc**2)
    return dist



class Ising():
    def __init__(self, Nr, Nc, J2, h, temp, l, config=[]):
        self.Nr = Nr
        self.Nc = Nc
        self.N = Nr*Nc
        self.beta = 1./temp
        self.J2 = J2
        self.h = h
        self.l = l

        self.Jij = self.BuildJij()

        if len(config) == 0:
            self.config = np.random.choice([1, -1], size=(self.N))
        else:
            self.config = config


    def BuildJij(self):
        Nr = self.Nr
        Nc = self.Nc
        N = self.N
        J2 = self.J2
        l = self.l

#         # Screening short-range interaction. Length scale l.
#         Jij = J2*np.exp(-GetTriDistances(Nr, Nc)/l)
#         for a in range(N):
#             Jij[a, a] = 0


#         # Nearest neighbor finite-range interaction J1, set to be 1.
#         for a in range(Nr):
#             for b in range(Nc):
#                 NN_list = GetNNTriOpen(a, b, Nr, Nc)
#                 for [c, d] in NN_list:
#                     Jij[a*Nc+b, c*Nc+d] += -1
        
        
        # Oscillating RKKY interaction. Length scale l is the Fermi wavelength.
        # r = np.maximum(0.8655*l, GetTriDistances(Nr, Nc))
        # Jij = - np.cos(r/l) / ((r/l)**3)
        # for a in range(N):
        #     Jij[a, a] = 0
        r = GetTriDistances(Nr, Nc)
        Jij = np.divide(-np.cos(2*r/l), r**3, out=np.zeros((N,N)), where=r!=0)
        
        return Jij

    def GetConfigs(self, Nwarmup, Ncycle, Lcycle):
        N = self.N
        self.MCSteps(Nwarmup)
        
        configs = np.zeros((Ncycle, N), dtype=int)

        for i in range(Ncycle):
            self.MCSteps(Lcycle)
            configs[i] = np.array(self.config)
            
        return configs

    def GetM(self):
        return abs(np.mean(self.config))
    
    def GetE(self):
        return np.dot(self.config, np.matmul(self.Jij, self.config))/self.N + self.h * np.mean(self.config)

    def GetFlipdE(self, i):

        # -2 for spin flipping
        # 2 in place of (s1, s2) <=> (s2, s1) symmetry
        return -2*self.config[i]*(2*np.sum(self.Jij[i]*self.config) + self.h)
    
    def GetS(self, omega, eps):
        N = self.N
        config = self.config
        beta = self.beta
        
        Em = np.zeros((N), dtype=float)
        
        for i in range(N):
            Em[i] = self.GetFlipdE(i)
        
        Sxx = np.mean(np.divide((1-np.exp(-beta*Em)), Em, out=np.full((N), beta), where=Em!=0) * (eps/((omega-Em)**2 + eps**2)))
        Sxy = -np.mean(config * np.divide((1-np.exp(-beta*Em)), Em, out=np.full((N), beta), where=Em!=0) * ((omega-Em)/((omega-Em)**2 + eps**2)))
        
        return [Sxx, Sxy]

    def GetSEm(self):
        N = self.N
        beta = self.beta
        
        Em = np.zeros((N), dtype=float)
        
        for i in range(N):
            Em[i] = self.GetFlipdE(i)
        return Em



    def MCSteps(self, Nstep):
        N = self.N
        beta = self.beta
        
        M_list = np.zeros((Nstep), dtype=int)
        
        for i in range(Nstep):
            a = np.random.randint(0, N)
            dE = self.GetFlipdE(a)
            
            if (dE < 0) or (random() < np.exp(-dE*beta)):
                self.config[a] *= -1

    def Measure(self, Nwarmup, Ncycle, Lcycle, omega=0., eps=0.05, print_time=60, meas_config=False, meas_M=True, meas_E=True, meas_S=False, meas_SEm=False):
        N = self.N

        print('Warming up')
        self.MCSteps(Nwarmup)
        
        config_list = np.zeros((Ncycle, N), dtype=int)
        M_list = np.zeros((Ncycle), dtype=float)
        E_list = np.zeros((Ncycle), dtype=float)
        S_list = np.zeros((2, Ncycle), dtype=float)
        SEm_list = np.zeros((Ncycle, N), dtype=float)
        
        t0 = time.time()
        t = time.time()

        print('Starting cycles')
        for i in range(Ncycle):
            self.MCSteps(Lcycle)
            
            if (time.time() - t) > print_time:
                perc = (i+1)/Ncycle
                eta = (time.time()-t0)*(Ncycle-i-1)/(i+1)
                print('%.4f complete, ETA %.2fm'%(perc, eta/60))
                t = time.time()

            if meas_config:
                config_list[i] = np.array(self.config)
            if meas_M:
                M_list[i] = self.GetM()
            if meas_E:
                E_list[i] = self.GetE()
            if meas_S:
                S_list[:, i] = self.GetS(omega, eps)
            if meas_SEm:
                SEm_list[i] = self.GetSEm()
            


        Out = {}
        if meas_config:
            Out['config'] = config_list
        if meas_M:
            Out['M'] = M_list
        if meas_E:
            Out['E'] = E_list
        if meas_S:
            Out['S'] = S_list
        if meas_SEm:
            Out['SEm'] = SEm_list.flatten()
        return Out




def GetEntropy(T_list, beta_E, interp_type='linear'):
    # Ent(beta) = Ent(beta = 0) + beta E(beta) - int(E(beta'), {beta', 0, beta})
    
    # Ent(beta = 0) = ln(2)
    # E(beta -> 0) = -(beta/2)\sum_{1\neq j}J_1j^2 - beta H^2g
    #                = -beta\sum_{1<j}J_1j^2 - beta H^2g


    beta = np.append(beta_E[0], 0)
    E = np.append(beta_E[1], 0)

    bsortind = np.argsort(beta)

    beta = beta[bsortind]
    E = E[bsortind]

    E_beta_fn = interp1d(beta, E, kind=interp_type, bounds_error=False, fill_value='extrapolate')

    Eint_beta_fn = interp1d(beta, cumtrapz(E, beta, initial=0), kind=interp_type, bounds_error=False, fill_value='extrapolate')


    beta_list = np.flip(np.append(1./T_list, 0))
    Ent_list = np.log(2) + beta_list*E_beta_fn(beta_list) - Eint_beta_fn(beta_list)

    return np.flip(Ent_list[1:])


def triPlot(ax, config, Nr, Nc):

    config = np.reshape(config, (Nr, Nc))    
    
    a = 1 # lattice constant
    r = a/np.sqrt(3)
    x = 0
    y = 0
    stagger = 1
    
    ax.set_aspect('equal')
    
    ax.set_xbound(-a/2, a*Nc)
    ax.set_ybound(a/np.sqrt(3), -a*(Nr*np.sqrt(3)/2-np.sqrt(3)/6))
    ax.set_axis_off()
    
    for nr in range(Nr):
        for nc in range(Nc):
            if config[nr, nc] > 0:
                color = 'red'
            else:
                color = 'blue'
            site = RegularPolygon((x, y), numVertices=6, radius=r, orientation=0, facecolor=color)
            ax.add_patch(site)
            
            x += a
        y -= a*np.sqrt(3)/2
        x = stagger*a/2
        stagger = 1 - stagger




def GetResistivity(Sx, Sy=[], Sx_err=[], Sy_err=[]):

    if Sy == []:
        Sy = 0*Sx
    if Sx_err == []:
        Sx_err = 0*Sx
    if Sy_err == []:
        Sy_err = 0*Sx


    Denom = Sx**2 + Sy**2
    Rx = Sx/Denom
    Ry = -Sy/Denom

    Rx_err = ((- Sx**2 + Sy**2)*Sx_err + (2*Sx*Sy)*Sy_err)/(Denom**2)
    Ry_err = ((- Sx**2 + Sy**2)*Sy_err + (2*Sx*Sy)*Sx_err)/(Denom**2)

    return Rx, Ry, Rx_err, Ry_err





def run_script(Nr, Nc, J2, T, h, l, Nwarmup, Ncycle, Lcycle, tag='', meas_config=False, meas_M=True, meas_E=True, meas_S=False, meas_SEm=False, eps=0.05):

    N = Nr*Nc
    
    if np.isscalar(J2):
        J2_list = np.array([J2])
    else:
        J2_list = np.array(J2)
    
    if np.isscalar(T):
        T_list = np.array([T])
    else:
        T_list = np.array(T)
        
    NT = len(T_list)

    if len(tag) == 0:
        now = datetime.now()
        tag = now.strftime("%y%m%d")

    t0 = time.time()

    print(T_list)
    
    for J2i in range(len(J2_list)):
        J2 = J2_list[J2i]


        config_array = np.zeros((NT, Ncycle, N), dtype=float)
        M_array = np.zeros((NT, Ncycle))
        E_array = np.zeros((NT, Ncycle))
        S_array = np.zeros((NT, 2, Ncycle))
        SEm_array = np.zeros((NT, Ncycle*N), dtype=float)
        
        config = []
        
        for Ti in range(NT):
            T = T_list[Ti]

            print('Beginning simulation for J2=%f, T=%f'%(J2, T))

            ising = Ising(Nr, Nc, J2, h, T, l, config)

            t = time.time()
            Out = ising.Measure(Nwarmup=Nwarmup, Ncycle=Ncycle, Lcycle=Lcycle, meas_config=meas_config, meas_M=meas_M, meas_E=meas_E, meas_S=meas_S, meas_SEm=meas_SEm)
            print('J2: %.2f, Temp: %.2f, Time: %.3fs'%(J2, T, time.time()-t))

            if meas_config:
                config_array[Ti] = Out['config']
            if meas_M:
                M_array[Ti] = Out['M']
            if meas_E:
                E_array[Ti] = Out['E']
            if meas_S:
                S_array[Ti] = Out['S']
            if meas_SEm:
                SEm_array[Ti] = Out['SEm']


            config = ising.config

        print('Total time: %.2fm'%((time.time()-t0)/60))

        

        wname = 'data/%s-l%.2f-J2%.2f-h%.2f-N%i.h5'%(tag, l, J2, h, N)

        if path.isfile(wname):
            now = datetime.now()
            wname = 'data/%s-%s-l%.2f-J2%.2f-h%.2f-N%i.h5'%(tag, now.strftime("%H%M%S"), l, J2, h, N)

        with h5py.File(wname, 'w') as f:
            f['T_list'] = T_list
            if meas_config:
                f['config_array'] = config_array
            if meas_M:
                f['M_array'] = M_array
            if meas_E:
                f['E_array'] = E_array
            if meas_S:
                f['S_array'] = S_array
            if meas_SEm:
                f['SEm_array'] = SEm_array



colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']