import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import matplotlib
np.set_printoptions(precision=5, suppress=True, linewidth=100)
plt.rcParams['figure.dpi'] = 150
import time
import logging
#import tenpy

#logging.basicConfig(filename='C:\\Users\\shakani3\\Sami\\dmrg\\tenpy\\dimerization\\HISTORY.log', level=logging.DEBUG)

logging.getLogger('matplotlib').setLevel(logging.WARNING) # only log warnings from matplotlib / tenpy 
logging.getLogger('tenpy').setLevel(logging.WARNING)
logging.basicConfig(filename='C:\\Users\\shakani3\\Sami\\dmrg\\tenpy\\dimerization\\HISTORY.log', level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

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

t_max, dt = 628., 0.0628

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

plot_params = { # parameters used to generate plots
    'H' : ['H1'],
    'R' : ['RD'],
    'N' : [80],
    'hz' : [0.,.1, .2, .3],
    'rz' : [0.],
}

# Hamiltonian class
class Hamiltonian(CouplingModel, MPOModel):
    def __init__(self, L, S=0.5, Jarr_cell=[1.], hz=0.):
        self.L = L
        spin = SpinSite(S=S, conserve="Sz")
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
            self.add_onsite(hz, u, 'Sz')
            
        
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


# Raman class

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
            
            self.data['corr_RR'].append(self.psi.overlap(temp_state)) # append <psi(t)|temp state> = <psi(t)|R|phi(t)> = <gs| e^iHt R e^-iHT R |gs>
    
    def timeEvolve(self): # only time evolves properly for nearest neighbor H
        engPsi = tebd.TEBDEngine(self.psi, self.H, self.tebd_params) # time evolve <g.s| e^iHt
        engPhi = tebd.TEBDEngine(self.phi, self.H, self.tebd_params) # time evolve e^-iHt R|g.s>
        
        while engPsi.evolved_time < self.t_max:
            self.measurement(engPsi)
            engPsi.run()
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
        
        logging.info("[{}, N={}, dt={}, t_max={}, TEBD order={}] DMRG {:.3f} s, TEBD {:.3f} s, FFT {:.3f} s".format(R_str, self.L, self.dt, self.t_max, self.tebd_params['order'], self.DMRG_time, self.TEBD_time, self.FFT_time))
        
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
        
    def PlotSpectrum(self, marker='.', removeElastic=True, ax=plt): # only plots magnitude of spectrum 
        if removeElastic:
            self.freq = self.freq[1:] # fft always puts elastic point as first
            self.i_fft = self.i_fft[1:]
        f, i = self.sort_data(self.freq, np.abs(self.i_fft)**2)
        '''
        self.freq, np.abs(self.i_fft)
        zipped_lists = zip(f,i)
        sorted_zipped = sorted(zipped_lists)
        i_s = np.array([el for _, el in sorted_zipped])
        '''
        ax.plot(f, i, marker)
        
    def CleanData(self, removeZeroT=True, removeReOffset=True, removeImOffset=True):
        if removeZeroT: # removes t = 0 point 
            self.data['t'] = self.data['t'][1:]
            self.data['corr_RR'] = self.data['corr_RR'][1:]
            
        if removeReOffset or removeImOffset: # removes average of Re part of <R(t)R(0)>
            cRe, cIm = int(removeReOffset), int(removeImOffset)
            mu = np.average(self.data['corr_RR']) # average of signal 
            self.data['corr_RR'] -= (cRe*mu.real + cIm*mu.imag) # removes real / imag parts of average from signal 
            
        self.freq, self.i_fft = self.corrFFT() # redo FFT 
        
        #f, i = self.sort_data(self.freq, np.abs(self.i_fft)**2)
        #plt.plot(f, i, marker)
    
    def GetPeak(self): # returns (w, imax) of inelastic peak 
        self.freq = self.freq[1:]
        self.i_fft = self.i_fft[1:]
        f, i = self.sort_data(self.freq, np.abs(self.i_fft)**2) # sorted data + removes elastic point
        
        index = i.argmax()
        return (f[index], i[index])
        
    def GetSpectralWeight(self):
        self.freq = self.freq[1:]
        self.i_fft = self.i_fft[1:]
        f, i = self.sort_data(self.freq, np.abs(self.i_fft)**2) # sorted data + removes elastic point
        
        index = i.argmax()
        i_max = i[index]
        
        k_l, k_r = 1, 1
        while i[index-k_l] > i_max*0.5:
            k_l += 1
        while i[index+k_r] > i_max*0.5:
            k_r += 1
        
        x1, y1 = np.array([f[index-k_l], f[index]]), np.array([i[index-k_l], i[index]])
        x2, y2 = np.array([f[index], f[index+k_r]]), np.array([i[index], i[index+k_r]])
        
        z1, z2 = np.polyfit(x1, y1, 1), np.polyfit(x2, y2, 1) # linear fit between just left & just right of peak frequency; highest degree first
        FWHM = ((0.5*i_max - z2[1])/z2[0]) - ((0.5*i_max - z1[1])/z1[0]) # A = (1/2)I_max * FWHM for Lorentzian
        return 0.5*i_max*FWHM
        
    def GetElasticPeak(self):
        return self.freq[0], np.abs(self.i_fft[0])**2
        
    def PlotSpectrumT(self): # plots time correlation function
        plt.plot(np.array(self.data['t']), np.array(self.data['corr_RR']))
        
    def SubplotSpectrum(self, zarg=False, lbound=-np.pi, rbound=2*np.pi, removeElastic=True, marker='.'): # shows & plots time correlation function, |I|, Re(I), Im(I) ; zarg=True plots Fourier phase 
        if removeElastic:
            self.freq = self.freq[1:] # fft always puts elastic point as first
            self.i_fft = self.i_fft[1:]
            
        fig, axs = plt.subplots(2,2)
        #axs[0,0].semilogy(*self.sort_data(self.freq, np.abs(self.i_fft)), '.') # magnitude of spectrum
        axs[0,0].plot(*self.sort_data(self.freq, np.abs(self.i_fft)**2), marker)
        
        
        if(zarg): # plotting complex phase 
            axs[0,1].plot( *self.sort_data(self.freq, np.angle(self.i_fft)/np.pi), marker)
        else:
            axs[0,1].plot( *self.sort_data(np.array(self.data['t']), np.array(self.data['corr_RR']).real), marker) # time correlation 
            axs[0,1].plot( *self.sort_data(np.array(self.data['t']), np.array(self.data['corr_RR']).imag), marker)
            axs[0,1].legend(['Re', 'Im'])
        axs[1,0].plot( *self.sort_data(self.freq, np.real(self.i_fft)), marker) # real part of spectrum 
        axs[1,1].plot( *self.sort_data(self.freq, np.imag(self.i_fft)), marker) # imaginary part of spectrum 
        
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
    
    def ShowSpectrum(self, legend_arr=None):
        plt.grid(which='major', axis='both')
        #plt.title('Raman Spectrum (N = {}) \n $(\epsilon_0 = {:.5f})$'.format(self.H.L, self.dmrg_info['E']))
        
        R_dict = {'RD':'$R_D$', 'RD3':'$R_D^{(3)}$', 'RD2':'$R_D^{(2)}$', 'R2':'$R_2$', 'R1':'$R_1$'}
        H_dict = {'HD':'$H_D$', 'HD3':'$H_D^{(3)}$', 'HD2':'$H_D^{(2)}$', 'H2':'$H_2$', 'H1':'$H_1$'}
        
        plt.title(H_dict[H_str]+ ', ' + R_dict[R_str] + ' Raman Spectrum (N = {})'.format(N))
        plt.xlabel('Frequency $(\omega/J)$')
        plt.ylabel('I (a.u.)')
        if not (legend_arr is None):
            plt.legend(legend_arr)
        plt.show()


# generate filenames from plot_params
traces = []
home_dir = 'C:\\Users\\shakani3\\Sami\\dmrg\\tenpy\\dimerization\\runs\\' # home directory of data 


for H in plot_params['H']:
    for R in plot_params['R']:
        for N in plot_params['N']:
            for hz in plot_params['hz']:
                for rz in plot_params['rz']:
                    filename = '{}-{}-n={}-hz={:.2f}-rz={:.2f}.csv'.format(H, R, N, hz, rz)
                    traces.append(filename)

# plot each file in trace array 
dummy_H, dummy_R = Hamiltonian(L = 4), Hamiltonian(L = 4) # dummy operators needed to initialize Raman object 
myRaman = Raman(dummy_H, dummy_R, t_max, dt, dmrg_params, tebd_params, 4) # dummy parameters to initialize Raman object 

'''
R_str='RDD'
myRaman.LoadSpectrum(home_dir+'HXX-RDD-n=60-hz=0.00-rz=0.00.csv')
myRaman.CleanData()
myRaman.SubplotSpectrum(lbound=-0.1, rbound=4., removeElastic=True, marker='-')
#plt.show()

'''

fig, ax = plt.subplots()

for trace in traces:
    fp = home_dir + trace
    myRaman.LoadSpectrum(fp)
    myRaman.PlotSpectrum('-', removeElastic=True, ax=ax)




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


#ax.xaxis.set_major_locator(MultipleLocator(0.5))
#ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#ax.xaxis.set_minor_locator(MultipleLocator(0.01))

#ax.yaxis.set_major_locator(MultipleLocator(1))
#ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
#ax.yaxis.set_minor_locator(MultipleLocator(0.1))

# plot aesthetics
plt.xlabel('$\\omega/J$')
plt.ylabel('$I$ (a.u.)')
plt.xlim(-0.1, 2)

# tickmarks
plt.grid(which='both')


# # plot title
title_str = ''

for param in plot_params:
    if len(plot_params[param]) == 1: # only one parameter (i.e. not changing)
        title_str += '{}={} '.format(param, plot_params[param][0])
plt.title(title_str)

# # plot legend
legend_arr = []
for H in plot_params['H']:
    for R in plot_params['R']:
        for N in plot_params['N']:
            for hz in plot_params['hz']:
                for rz in plot_params['rz']:
                    H_str, R_str, N_str, hz_str, rz_str = '', '', '', '', ''
                    if len(plot_params['H']) > 1:
                        H_str = H
                    if len(plot_params['R']) > 1:
                        R_str = R
                    if len(plot_params['N']) > 1:
                        N_str = 'N={}'.format(N)
                    if len(plot_params['hz']) > 1:
                        hz_str = 'hz={:.2f}'.format(hz)
                    if len(plot_params['rz']) > 1:
                        rz_str = 'rz={:.2f}'.format(rz)
                    trace_name = ' '.join([H_str, R_str, N_str, hz_str, rz_str])
                    legend_arr.append(trace_name)

plt.legend(legend_arr)

plt.show() 