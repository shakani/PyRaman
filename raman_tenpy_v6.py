import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib
np.set_printoptions(precision=5, suppress=True, linewidth=100)
plt.rcParams['figure.dpi'] = 150
import time
import logging
import os
#import tenpy

#logging.basicConfig(filename='C:\\Users\\shakani3\\Sami\\dmrg\\tenpy\\dimerization\\HISTORY.log', level=logging.DEBUG)

logging.getLogger('matplotlib').setLevel(logging.WARNING) # only log warnings from matplotlib / tenpy 
logging.getLogger('tenpy').setLevel(logging.WARNING)
logging.basicConfig(filename='HISTORY.log', level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

from tenpy.models.xxz_chain import XXZChain
from tenpy.models.spins import SpinChain
from tenpy.models.spins_nnn import SpinChainNNN2

from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.algorithms.exact_diag import ExactDiag
from tenpy.algorithms import tebd

from tenpy.networks.site import SpinSite
from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingModel, NearestNeighborModel, MPOModel

import sys

#tenpy.tools.misc.setup_logging(to_stdout="INFO")

startTime = time.time()

N = int(sys.argv[1])
H_str = sys.argv[2]
R_str = sys.argv[3]
h_field = 0.00
r_field = 0.50

#Jarr = [1.,-1.] * int(N/2)
#Jarr = Jarr[:-1]
t_max, dt = 628., 0.0628
# t_max * domega = 2pi 
# dt * (omega_max - omega_min) = 2pi  

dmrg_params = {
    'mixer': True,
    'max_E_err': 1.e-10,
    'trunc_params': {
        'chi_max': 100,
        'svd_min': 1.e-10,
    }
    #'verbose':True,
    #'combine':True
}

tebd_params = {
    'N_steps': 1,
    'dt': dt,
    'order': '4_opt', # order of Suzuki-Trotter decomposition
    'trunc_params': {'chi_max':100, 'svd_min':1.e-12}
}


def JarrPattern(Jarr, n=1): # returns array which tiles Jarr onto lattice size N
    if len(Jarr) >= N:
        return Jarr[:N-n]
    else:
        Jarr_temp = Jarr * N
        return JarrPattern(Jarr_temp, n)
# n parameterizes J1 or J2

def flatten(arr): # flattens array  
    return [item for sublist in arr for item in sublist]

class Hamiltonian(CouplingModel, MPOModel):
    def __init__(self, L, S=0.5, Jarr_cell=[1.], hz=0.):
        self.L = L
        #spin = SpinSite(S=S, conserve="Sz")
        spin = SpinSite(S=S, conserve="None")
        self.Jarr = self.JarrPattern(Jarr_cell)
        #print(len(self.Jarr))
        # the lattice defines the geometry
        lattice = Chain(L, spin, bc="open", bc_MPS="finite")
        CouplingModel.__init__(self, lattice)
        # add terms of the Hamiltonian
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(self.Jarr, u1, 'Sz', u2, 'Sz', dx)
            self.add_coupling([0.5 * J for J in self.Jarr], u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            
        for u in range(len(self.lat.unit_cell)): # adds static field in z direction 
            self.add_onsite(-1*hz, u, 'Sz')
        
        
        #for u1, u2, dx in self.lat.pairs['next_nearest_neighbors']:
        #    self.add_coupling(Jarr[1], u1, 'Sz', u2, 'Sz', dx)
        #    self.add_coupling(0.5 * Jarr[1], u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            
        # finish initialization
        # generate MPO for DMRG
        MPOModel.__init__(self, lattice, self.calc_H_MPO())
        # generate H_bond for TEBD
        NearestNeighborModel.__init__(self, lattice, self.calc_H_bond())
        
    def JarrPattern(self, JJarr):
        if len(JJarr) >= self.L:
            return JJarr[:self.L-1]
        else:
            Jarr_temp = JJarr * self.L 
            return self.JarrPattern(Jarr_temp)
            
class XYModel(CouplingModel, MPOModel):
    def __init__(self, L, S=0.5, Jarr_cell=[1.], hz=0.):
        self.L = L
        #spin = SpinSite(S=S, conserve="Sz")
        spin = SpinSite(S=S, conserve="None")
        self.Jarr = self.JarrPattern(Jarr_cell)
        #print(len(self.Jarr))
        # the lattice defines the geometry
        lattice = Chain(L, spin, bc="open", bc_MPS="finite")
        CouplingModel.__init__(self, lattice)
        # add terms of the Hamiltonian
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']: 
            #self.add_coupling(self.Jarr, u1, 'Sz', u2, 'Sz', dx) # no zz coupling
            self.add_coupling([0.5 * J for J in self.Jarr], u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            
        for u in range(len(self.lat.unit_cell)): # adds static field in z direction 
            self.add_onsite(-1*hz, u, 'Sz')
        
        
        #for u1, u2, dx in self.lat.pairs['next_nearest_neighbors']:
        #    self.add_coupling(Jarr[1], u1, 'Sz', u2, 'Sz', dx)
        #    self.add_coupling(0.5 * Jarr[1], u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            
        # finish initialization
        # generate MPO for DMRG
        MPOModel.__init__(self, lattice, self.calc_H_MPO())
        # generate H_bond for TEBD
        NearestNeighborModel.__init__(self, lattice, self.calc_H_bond())
        
    def JarrPattern(self, JJarr):
        if len(JJarr) >= self.L:
            return JJarr[:self.L-1]
        else:
            Jarr_temp = JJarr * self.L 
            return self.JarrPattern(Jarr_temp)

class J1J2Model(CouplingModel, MPOModel):
    def __init__(self, L=N, S=0.5, J1arr_cell=[1.], J2arr_cell=[1.]):
        self.L = N
        spin = SpinSite(S=S, conserve="Sz")
        J1arr = JarrPattern(J1arr_cell)
        J2arr = JarrPattern(J2arr_cell, n=2)
        lattice = Chain(L, spin, bc="open", bc_MPS="finite")
        CouplingModel.__init__(self, lattice)
        
        # couplings
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(J1arr, u1, 'Sz', u2, 'Sz', dx)
            self.add_coupling([0.5 * J1 for J1 in J1arr], u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
        for v1, v2, dy in self.lat.pairs['next_nearest_neighbors']:
            self.add_coupling(J2arr, v1, 'Sz', v2, 'Sz', dy)
            self.add_coupling([0.5 * J2 for J2 in J2arr], v1, 'Sp', v2, 'Sm', dy, plus_hc=True)
        
        self.grouped_sites = self.group_sites(2)
        
        MPOModel.__init__(self, lattice, self.calc_H_MPO())
        NearestNeighborModel.__init__(self, lattice, self.calc_H_bond())

class Raman():
    def __init__(self, H_model, R_model, t_max, dt, dmrg_params, tebd_params, chain_length, nnn=False):
        self.H = H_model
        self.R = R_model
        self.L = chain_length
        self.R_nn = None
        self.t_max = t_max
        self.dt = dt
        self.dmrg_params = dmrg_params
        self.tebd_params = tebd_params
        self.tebd_params['dt'] = dt # makes sure tebd_params & Raman class agree 
        
        self.nnn = nnn # flag for if we're using next nearest neighbors
        
        psi_arr = [[['up'],['down']][int(i%2)] for i in range(self.L)]
        self.psi = MPS.from_lat_product_state(H_model.lat, psi_arr)
        #self.psi = MPS.from_lat_product_state(H_model.lat, [['up'], ['down']])
        self.dmrg_info = {'E': 0.}
        '''
        self.dmrg_info = dmrg.run(self.psi, self.H, self.dmrg_params) # runs DMRG on to turn psi to ground state
        self.phi = self.psi.copy() # copies gs to phi
        self.R.H_MPO.apply_naively(self.phi) # |phi> = R|g.s>
        '''
        self.data = {'t':[], 'corr_RR':[]} #dict([(k, [])] for k in ['t', 'corr_RR'])
        
        self.freq = None
        self.i_fft = None
    
    def NNNgrouping(self): # group wavefunction and models to work as NN models
        self.psi.group_sites(n=2)
        self.phi.group_sites(n=2)
        self.H.group_sites(n=2)
        self.R.group_sites(n=2)
        self.R_nn = NearestNeighborModel.from_MPOModel(self.R) # creates nn model from nnn Raman operator
    
    def getGroundState(self):
        self.dmrg_info = dmrg.run(self.psi, self.H, self.dmrg_params) # runs DMRG on to turn psi to ground state
        self.phi = self.psi.copy() # copies gs to phi
        if self.nnn:
            self.NNNgrouping() # groups sites if nnn
            #self.R_nn.H_MPO.apply_naively(self.phi)
        self.R.H_MPO.apply_naively(self.phi) # |phi> = R|g.s>
    
    def measurement(self, eng):
        self.data['t'].append(eng.evolved_time)
        
        temp_state = self.phi.copy() # copies RHS of matrix element |phi(t)>
        if self.nnn:
            self.R_nn.H_MPO.apply_naively(temp_state)
            self.data['corr_RR'].append(self.psi.overlap(temp_state))
        else:
            self.R.H_MPO.apply_naively(temp_state) # act with R on temp state R|phi(t)> = R e^-iHt R|g.s.>
            #self.R.H_MPO.apply(temp_state)
            
            self.data['corr_RR'].append(np.exp(eng.evolved_time * 1j * float(self.dmrg_info['E']))*self.psi.overlap(temp_state)) # append <psi(t)|temp state> = <psi(t)|R|phi(t)> = <gs| e^iHt R e^-iHT R |gs>

    def getSz(self):
        '''if hh == 0 and JJ == 0:
            psi_arr = [[['up'],['down']][int(i%2)] for i in range(self.L)]
        elif abs(hh/JJ) > 0 or JJ < 0: # for large field or FM interaction, guess spin polarized 
            if hh >= 0: # up field
                psi_arr = [[['up'],['up']][int(i%2)] for i in range(self.L)]
            else: # down field
                psi_arr = [[['down'],['down']][int(i%2)] for i in range(self.L)]
                
        else: # AFM interaction dominated, guess Neel state
            psi_arr = [[['up'],['down']][int(i%2)] for i in range(self.L)]'''
        
        psi_arr = [[['up'],['down']][int(i%2)] for i in range(self.L)]    
        self.psi = MPS.from_lat_product_state(self.H.lat, psi_arr) # apply guess for GS 
        self.getGroundState() # finds gs 
        
        Sz_measured = self.psi.expectation_value('Sz')
        return Sz_measured # returns Sz
    
    def timeEvolve(self): # only time evolves properly for nearest neighbor H
        #engPsi = tebd.TEBDEngine(self.psi, self.H, self.tebd_params) # time evolve <g.s| e^iHt
        engPhi = tebd.TEBDEngine(self.phi, self.H, self.tebd_params) # time evolve e^-iHt R|g.s>
        
        while engPhi.evolved_time < self.t_max:
            self.measurement(engPhi)
            #engPsi.run()
            engPhi.run()
            #self.measurement(engPsi)
    
    
    def corrFFT(self):
        t, i = np.array(self.data['t']), np.array(self.data['corr_RR'])
        i_fft = np.fft.fft(i) / len(i)
        #freq = np.fft.fftfreq(np.size(i_fft), d=self.dt) # maybe correct FT convention here 
        freq = np.fft.fftfreq(np.size(i_fft), d=self.dt) # maybe correct FT convention here 
        return (-2*np.pi)*freq, i_fft
    
    def Spectrum(self, nnn=False):
        t0 = time.time() # starting time 
        self.getGroundState()
        t1 = time.time()
        self.timeEvolve()
        t2 = time.time()
        freq, i_fft = self.corrFFT()
        self.freq, self.i_fft = freq, i_fft
        t3 = time.time()
        
        self.DMRG_time, self.TEBD_time, self.FFT_time = t1-t0, t2-t1, t3-t2
        
        logging.info("[{}, {}, N={}, dt={}, t_max={}, TEBD order={}] DMRG {:.3f} s, TEBD {:.3f} s, FFT {:.3f} s".format(H_str, R_str, self.L, self.dt, self.t_max, self.tebd_params['order'], self.DMRG_time, self.TEBD_time, self.FFT_time))
        
    def SaveSpectrum(self, filepath):
        arr = np.array([self.data['t'], self.data['corr_RR']])
        arr = np.transpose(arr)
        np.savetxt(filepath, arr, delimiter=',')
        
    def LoadSpectrum(self, filepath):
        arr = np.transpose(np.genfromtxt(filepath, delimiter=',', dtype=str))
        '''
        self.freq = np.array([np.real(_) for _ in arr[0]])
        self.i_fft = np.array([np.complex(_) for _ in arr[1]])
        '''
        #self.data['t'] = np.fromiter(map(np.complex, arr[0]), dtype=complex)
        #self.data['corr_R'] = np.fromiter(map(np.complex, arr[1]), dtype=complex)
        self.data['t'] = list(map(np.complex, arr[0]))
        self.data['corr_RR'] = list(map(np.complex, arr[1]))
        self.freq, self.i_fft = self.corrFFT()
        '''if removeElastic:
            self.freq = self.freq[1:] # fft always puts elastic point as first
            self.i_fft = self.i_fft[1:]'''
    
    def sort_data(self, arr1, arr2): # returns arr1 and arr2 sorted based on ordering of arr1 
        zipped_lists = zip(arr1, arr2)
        sorted_zipped = sorted(zipped_lists)
        arr1_s = sorted(arr1)
        arr2_s = [el for _, el in sorted_zipped]
        return np.array(arr1_s), np.array(arr2_s)
    
    def Resample(self, dt_new=dt, t_max_new=t_max): # downsamples time domain and takes shorter series (i.e. chops off long time behavior)
        rate = int(dt_new/self.dt)
        self.data['t'] = self.data['t'][::rate]
        self.data['corr_RR'] = self.data['corr_RR'][::rate] # down sampling done
        self.data['t'] = [_ for _ in self.data['t'] if abs(_) < t_max_new]
        self.data['corr_RR'] = self.data['corr_RR'][:len(self.data['t'])] # chopping 
        # chop off data?
        self.dt, self.t_max = dt_new, t_max_new 
        self.freq, self.i_fft = self.corrFFT()
        
    def PlotSpectrum(self, marker='.'): # only plots magnitude of spectrum 
        f, i = self.sort_data(self.freq, np.abs(self.i_fft))
        '''
        self.freq, np.abs(self.i_fft)
        zipped_lists = zip(f,i)
        sorted_zipped = sorted(zipped_lists)
        i_s = np.array([el for _, el in sorted_zipped])
        '''
        plt.plot(f, i, marker)
        
        
    def PlotSpectrumT(self): # plots time correlation function
        plt.plot(np.array(self.data['t']), np.array(self.data['corr_RR']))
        
    def SubplotSpectrum(self, zarg=False, lbound=-np.pi, rbound=2*np.pi, removeElastic=True): # shows & plots time correlation function, |I|, Re(I), Im(I) ; zarg=True plots Fourier phase 
        if removeElastic:
            self.freq = self.freq[1:] # fft always puts elastic point as first
            self.i_fft = self.i_fft[1:]
            
        fig, axs = plt.subplots(2,2)
        #axs[0,0].semilogy(*self.sort_data(self.freq, np.abs(self.i_fft)), '.') # magnitude of spectrum
        axs[0,0].plot(*self.sort_data(self.freq, np.abs(self.i_fft)**2), '.')
        
        
        if(zarg): # plotting complex phase 
            axs[0,1].plot( *self.sort_data(self.freq, np.angle(self.i_fft)/np.pi), '.')
        else:
            axs[0,1].plot( *self.sort_data(np.array(self.data['t']), np.array(self.data['corr_RR']).real), '.') # time correlation 
            axs[0,1].plot( *self.sort_data(np.array(self.data['t']), np.array(self.data['corr_RR']).imag), '.')
            axs[0,1].legend(['Re', 'Im'])
        axs[1,0].plot( *self.sort_data(self.freq, np.real(self.i_fft)), '.') # real part of spectrum 
        axs[1,1].plot( *self.sort_data(self.freq, np.imag(self.i_fft)), '.') # imaginary part of spectrum 
        
        # formatting 
        axs[0,0].set_title(R_str + ' Power Spectrum $(N = {})$'.format(N))
        #axs[0,0].set_xlabel('Frequency $(\omega)$')
        axs[0,0].set_ylabel('I (a.u.)')
        axs[0,0].grid(which='major', axis='both')
        
        if(zarg):
            axs[0,1].set_title('Arg($I(\omega)$)')
            #axs[0,1].set_xlabel('Frequency')
            axs[0,1].set_ylabel('Phase ($\pi$ rad)')
        else:
            axs[0,1].set_title(R_str + ' Raman Time Correlation')
            axs[0,1].set_xlabel('Time $(t)$')
            #axs[0,1].set_ylabel('$\langle R(t)R(0) \rangle$ (a.u.)')
        axs[0,1].grid(which='major', axis='both')
        
        axs[1,0].set_title('Re($I(\omega)$)')
        axs[1,0].set_xlabel('Frequency $(\omega)$')
        axs[1,0].set_ylabel('Re($I$) (a.u.)')
        axs[1,0].grid(which='major', axis='both')
        
        axs[1,1].set_title('Im($I(\omega)$)')
        axs[1,1].set_xlabel('Frequency $(\omega)$')
        axs[1,1].set_ylabel('Im($I$) (a.u.)')
        axs[1,1].grid(which='major', axis='both')
        
        axs[0,0].set_xlim(lbound, rbound)
        axs[1,0].set_xlim(lbound, rbound)
        axs[0,1].set_xlim(lbound, rbound)
        axs[1,1].set_xlim(lbound, rbound)
        
        plt.tight_layout()
        plt.show()
    
    def ShowSpectrum(self):
        plt.grid(which='major', axis='both')
        #plt.title('Raman Spectrum (N = {}) \n $(\epsilon_0 = {:.5f})$'.format(self.H.L, self.dmrg_info['E']))
        plt.title(R_str + ' Raman Spectrum (N = {})'.format(N))
        plt.xlabel('Frequency $(\omega)$')
        plt.ylabel('I (a.u.)')
        plt.show()



class Domain(): # domain walls can't be at first two or last two lattice sites
    def __init__(self, dWalls):
        self.dWalls = dWalls
        self.bonds = [1.]*(N-1)
        #self.bonds = self.bonds[:N-1] # bond initialization as alternating
        for d in self.dWalls:
            self.placeSpin(d-1)
    
    def placeSpin(self, k): # isolates spin at lattice site k 
        self.bonds[k-1], self.bonds[k] = 0.,0.
        
    def dimerize(self, j1, j2): # dimerizes between two lattice sites
        k1, k2 = min(j1, j2), max(j1, j2) # k1 < k2
        for j in range(1,k2-k1): 
            self.bonds[k1+j-1] = float(j % 2) # loop through bonds 
        
    def jarr(self):
        return self.bonds 

class singleDomain():
    def __init__(self, k): # isolated spin at lattice site k
        self.k = k
        if k%2:
            self.bonds = [1.,0.]*N
        else:
            self.bonds = [0.,1.]*N
        self.bonds = self.bonds[:N-1]
        
        self.bonds[k-1], self.bonds[k] = 0., 0.
        
        for i in range(len(self.bonds)):
            if i >= k: # k odd; for k even need i > k
                self.bonds[i] = self.flip(self.bonds[i])
    
    def jarr(self):
        return self.bonds
    
    def flip(self, site):
        return float((site+1)%2)


def domainToJarr(n, m): # for n sites and m domains of alternating +1/-1 couplings 
    if m == 1:
        return [1., -1.] # just RD 
    elif m == 2: # only for odd numbered sites
        k = int((n-1)/2)
        domain = [1., -1.] * int(k/2)
        domain = domain[:-1]
        return domain + [-1., -1.] + domain
        
    elif m == 3:
        ed = n//3 - (n//3)%2 # number of sites in outer domains 
        mid = n - 2 - 2*ed # number of sites in middle domain; domains comprise n-2 sites total since there are 2 domain walls 
        eddom = [1., -1.] * int(ed/2)
        middom = [1., -1.] * int(mid/2)
        eddom, middom = eddom[:-1], middom[:-1] # alterates +1 -1 bond strength 
        return eddom + [-1., -1.] + middom + [-1., -1.] + eddom 

def domainToJarr4(n): # for n sites returns +1/-1 couplings and splits domains (close to) sizes m | 2m | m
    a = np.array([(-1)**i for i in range(n-1)])
    b = np.array([1 for i in range(n-1)])
    b[n//4 : (n - n//4)] *= -1
    return np.multiply(a, b)
        
def domainToJarrClose(n): # creates 2 defects in Raman operator that are close together as +1/-1 couplings
    arr = [(-1)**i for i in range(n-1)] # 1 -1 1 -1 ...
    z = int(n/2)
    arr[z] *= -1 # flip these 3 bonds near the middle to create two domain walls
    arr[z-1] *= -1
    arr[z-2] *= -1
    return arr 

def simpleJarr():
    return [0.00031641574809688594, -0.0003043807504335147, 0.00028466034697937243, -0.000257752464435432, 0.0002243365084343615, -0.00018525620901293035, 0.00014149831700557987, -9.416768926067017e-05, 4.445939176063717e-05, 6.37147497678899e-06, -5.704146632832884e-05, 0.00010627119966417805, -0.0001528176578844755, 0.0001955055747510092, -0.00023325710955599867, 0.0002651190618628608, -0.0002902869391653655, 0.000308125269773248, -0.00031818364803780586, 0.0003202081067865091, -0.0003141475298204249, 0.000300154942563284, -0.0002785836482741812, 0.00024997830738189775, -0.00021506118518155318, 0.00017471391512984056, -0.00012995523820317063, 8.191528038470577e-05, -3.180701775614851e-05, -1.9104350318801405e-05, 6.95333466131447e-05, -0.00011820667345746117, 0.00016389536263528996, -0.0002054458059901363, 0.00024180888324250332, -0.0002720664515594821, 0.0002954545280333943, -0.0003113825797277392, 0.0003194484342299939, -0.0003194484342299939, 0.0003113825797277392, -0.0002954545280333943, 0.0002720664515594821, -0.00024180888324250332, 0.0002054458059901363, -0.00016389536263528996, 0.00011820667345746117, -6.95333466131447e-05, 1.9104350318801405e-05, 3.180701775614851e-05, -8.191528038470577e-05, 0.00012995523820317063, -0.00017471391512984056, 0.00021506118518155318, -0.00024997830738189775, 0.0002785836482741812, -0.000300154942563284, 0.0003141475298204249, -0.0003202081067865091, 0.00031818364803780586, -0.000308125269773248, 0.0002902869391653655, -0.0002651190618628608, 0.00023325710955599867, -0.0001955055747510092, 0.0001528176578844755, -0.00010627119966417805, 5.704146632832884e-05, -6.37147497678899e-06, -4.445939176063717e-05, 9.416768926067017e-05, -0.00014149831700557987, 0.00018525620901293035, -0.0002243365084343615, 0.000257752464435432, -0.00028466034697937243, 0.0003043807504335147, -0.00031641574809688594, 0.00032046146450889285]

def realH(n, dJ):
    bondarr = list(map(lambda x: (x+1.)/2, domainToJarr(n, 3)))
    for j in range(len(bondarr)-1):
        if bondarr[j] == 0 and bondarr[j+1] == 0: # look for defects in bond array
            bondarr[j], bondarr[j+1] = 2, 3 # label them to be changed

    f = lambda x: [1., 1., 1., 1.+dJ][int(x)] # maps 0 -> 1, 1 -> 1, 2 -> 0, and 3 -> 1 + dJ
    bondarr = list(map(f, bondarr))

    return bondarr

def realRD3(n, dJ):
    bondarr = list(map(lambda x: (x+1.)/2, domainToJarr(n, 3)))
    for j in range(len(bondarr)-1):
        if bondarr[j] == 0 and bondarr[j+1] == 0: # look for defects in bond array
            bondarr[j+1] = dJ # changed

    return bondarr


#H = Hamiltonian(Jarr_cell=[1.])

H_params = dict(
        L=N,
        S=0.5,
        Jx=1.,
        Jy=1.,
        Jz=1.,
        hz=0.,
        bc_MPS='finite',
        conserve='Sz',
    )
R_params = dict(
        L=N,
        S=0.5,
        Jx=0.,
        Jy=0.,
        Jz=0.,
        Jxp=1.,
        Jyp=1.,
        Jzp=1.,
        hz=0.,
        bc_MPS='finite',
        conserve='Sz',
    )
    
H = SpinChain(H_params) ## nn

#narr = [21, 41, 61, 81, 101, 121, 161, 201]
narr = [80]
harr = [.5]
rarr = [0.]

if __name__ == "__main__":
        for nn in narr:
            for hh in harr:
                for rr in rarr:
                    H_params['L'], R_params['L'] = nn, nn
                    H_params['hz'], R_params['hz'] = hh, rr
                    
                    if H_str == 'H1':
                        '''H_params['hz'] = hh # assigns field
                        H = SpinChain(H_params) # H is nearest neighbor Heisenberg'''
                        H = Hamiltonian(L=nn, Jarr_cell=[1.], hz=hh) # mine actually converges to gs with DMRG 
                    elif H_str == 'HD':
                        H = Hamiltonian(L=nn, Jarr_cell=[1.,0.], hz=hh) # H is fully dimerized; zero field
                    elif H_str == 'HD3':
                        starting_jarr = domainToJarr(nn, 3)
                        my_jarr = [0.5*(j+1) for j in starting_jarr] # maps -1 and 1 to 0 and 1 resp.
                        H = Hamiltonian(L=nn, Jarr_cell=my_jarr, hz=hh)
                    elif H_str == 'HDw':
                        H = Hamiltonian(L=nn, Jarr_cell=[1.,0.1], hz=hh) # H has weak bonds of 10%
                    elif H_str == 'HD3w':
                        starting_jarr = domainToJarr(nn, 3)
                        my_jarr = [0.5*(0.9*j+1.1) for j in starting_jarr] # maps -1 and 1 to 0.1 and 1 resp.
                        H = Hamiltonian(L=nn, Jarr_cell=my_jarr, hz=hh)
                    elif H_str == 'HXX':
                        H = XYModel(L=nn, Jarr_cell=[1.], hz=hh) # pure XY model 
                    elif H_str == 'Hreal':
                        real_jarr = realH(nn, 0.3) # real Hamiltonian defects
                        H = Hamiltonian(L = nn, Jarr_cell = real_jarr, hz=hh)
                    
                    if R_str == 'RD':
                        R = Hamiltonian(L=nn, Jarr_cell=[1.,-1.], hz=rr) # dimerized Raman operator
                    elif R_str == 'RD2':
                        R = Hamiltonian(L=nn, Jarr_cell=domainToJarr(nn, 2), hz=rr)
                    elif R_str == 'RD3':
                        R = Hamiltonian(L=nn, Jarr_cell=domainToJarr(nn, 3), hz=rr)
                    elif R_str == 'R2':
                        R_params['hz'] = rr # assigns field 
                        R = SpinChainNNN2(R_params) # R2 Raman operator 
                    elif R_str == 'R1':
                        R = Hamiltonian(L=nn, Jarr_cell=[1.], hz=rr) # Heisenberg operator
                    elif R_str == 'RDD':
                        R = Hamiltonian(L=nn, Jarr_cell=[1.,0.], hz=rr) # dimerized Raman operator
                    elif R_str == 'RD3-close':
                        R = Hamiltonian(L=nn, Jarr_cell=domainToJarrClose(nn), hz=rr) # 2 defects close together 
                    elif R_str == 'RXXD':
                        R = XYModel(L=nn, Jarr_cell=[1., 0.], hz=rr)
                    elif R_str == 'RXXD3':
                        starting_jarr = domainToJarr(nn, 3)
                        my_jarr = [0.5*(j+1) for j in starting_jarr] # maps -1 and 1 to 0 and 1
                        R = XYModel(L=nn, Jarr_cell=my_jarr, hz=rr)
                    elif R_str == 'RD4': # 3 domain Raman operator with domain sizes m | 2m | m, such that n = 4m
                        R = Hamiltonian(L=nn, Jarr_cell=domainToJarr4(nn), hz=rr)
                    elif R_str == 'Rsimp': 
                        R = Hamiltonian(L=nn, Jarr_cell=simpleJarr(), hz=rr)
                    elif R_str == 'RD3real':
                        real_jarr_r = realRD3(nn, 0.5)
                        R = Hamiltonian(L = nn, Jarr_cell = real_jarr_r, hz=rr)

                    myRaman = Raman(H, R, t_max, dt, dmrg_params, tebd_params, chain_length=nn)
                    myRaman.Spectrum()
                    #myRaman.SaveSpectrum('C:\\Users\\shakani3\\Sami\\dmrg\\tenpy\\dimerization\\runs\\{}-{}-n={}-hz={:.2f}-rz={:.2f}.csv'.format(H_str,R_str,nn,hh,rr))

                    if not os.path.exists('runs'): # create directory for output if needed
                        os.mkdir('runs')

                    if os.name == 'nt': # windows 
                        myRaman.SaveSpectrum('runs\\{}-{}-n={}-hz={:.3f}-rz={:.2f}.csv'.format(H_str,R_str,nn,hh,rr))

                    elif os.name == 'posix': # linux
                        myRaman.SaveSpectrum('runs/{}-{}-n={}-hz={:.3f}-rz={:.2f}.csv'.format(H_str,R_str,nn,hh,rr))

                    #m = myRaman.getSz()
                    #print('hz = {}, m = {}'.format(hh, float(sum(m)/myRaman.L)))
