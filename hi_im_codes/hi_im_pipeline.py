"""HI IM pipeline for MeerKAT, BINGO, FAST, SKAI"""

import glob
import numpy as np
import h5py
import healpy as hp
import matplotlib.pyplot as plt
import pylab
import matplotlib as mpl
mpl.rcParams['axes.formatter.useoffset'] = False

class Sky_Maps(object):
    
    def __init__(self, maps, freq_channels, cl_length, pol, counter, nside, theta_min, theta_max, name):

        self.maps = maps
        self.freq_channels = freq_channels
        self.cl_length = cl_length
        self.pol = pol
        self.counter = counter
        self.nside = nside
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.name = name
    @property
    def sorted_files(self):
        filenames = self.maps 
        filenames = filenames[:]
        return filenames

    @property
    def ang_power_spectra(self):
        #ngular power spectra
        cls = np.zeros(shape = (len(self.maps), self.freq_channels, self.cl_length)) 
        j = self.counter 
        
        for f in self.sorted_files:
            print ("File number: {}".format(j))
            files = h5py.File(f, 'r')
            info = files['index_map']
            freq = info['freq'][:]
            map_dset = files['map']

            # row and column sharing
            
            for i in range(freq.shape[0]):
                #fig = plt.figure(figsize=(15.0, 4.0))
                #Mask maps
                map_I = map_dset[i,self.pol] #Unmasked map
                mask = map_dset[i,self.pol].astype(np.bool)
                map_I_masked = hp.ma(map_I) #loads a map as a masked array
                pixel_theta, pixel_phi = hp.pix2ang(self.nside, np.arange(hp.nside2npix(self.nside)))
                #print(pixel_theta, pixel_phi)
                mask[pixel_theta < self.theta_min*np.pi/180] = 0 
                mask[pixel_theta > self.theta_max*np.pi/180] = 0
                map_I_masked.mask = np.logical_not(mask)
                fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15.0, 4.0))
                ax3.xaxis.labelpad = 15
                plt.subplots_adjust(top = 2.0)
                #ax1 = fig.add_subplot(1, 2, 1)
                #ax2 = fig.add_subplot(1, 2, 2)
                #ax3 = fig.add_subplot(1, 2, 3)
                if i == 0:
                    fig.suptitle('{}'.format(self.name))
                plt.axes(ax3)
                hp.mollview(map_I_masked.filled(), coord=['C'], \
                unit='K', norm='hist', xsize=2000, return_projected_map=True, hold=True)
                #plt.savefig('/home/elimboto/Desktop/Forecast_Codes/hi_im_pipeline/BINGO_strip%s.pdf'% ("%.4f" % round(i, 4)), format='pdf', bbox_inches='tight')
                #hp.graticule()
                #Remove all the masked pixels and return a standard array 
                compressed_map = map_I_masked.compressed()
                print("The number of UNSEEN pixels:", 12*self.nside**2 - len(compressed_map))
                ax1.hist(compressed_map, bins = 50, color = 'green', facecolor = 'lightblue')
                for label in ax1.xaxis.get_ticklabels():
                    label.set_rotation(45)
                #ax1.set_title('{} Hist, freq={}'.format(self.name, ("%.4f" % round(freq[i][0], 4))), fontsize=20)
                #plt.show()
                cl = hp.sphtfunc.anafast(map_I_masked.filled(), lmax = None) 
                #ax2.set_title("freq = {}".format("%.4f" % round(freq[i][0], 4)), fontsize=20)
                ax2.set_xlabel(r"$\ell$")
                ax2.loglog(cl, label = "freq = %s" % ("%.4f" % round(freq[i][0], 4)))
                ax2.legend(loc = 1 )
                ax2.set_ylim(ymax=9.e-10, ymin=1.e-15)  
                ax2.set_xlim(xmax=10e3, xmin=0)
                fig.tight_layout()
                fig.savefig('/home/elimboto/Desktop/Forecast_Codes/hi_im_pipeline/BINGO_cl%s.pdf'% ("%.4f" % round(i, 4)), format='pdf', bbox_inches='tight')
                #plt.show()
                cls[j][i] = cl
            j += 1
        return cls
       
    @property
    def plot_cls(self):
        #Compute Average Cls if needed
        fig3 = plt.figure(figsize=(10.0, 3.0))
        #fig = mp.figure()
        ax3 = fig3.add_subplot(111) #, aspect='equal'
        ax3.set_aspect('auto')
        cls_mean = np.mean(self.ang_power_spectra, axis = 0)
        freqs = np.linspace(960, 1260, 30+1, endpoint=True)
        freqs= (freqs[1:] + freqs[:-1])/2.0
        for k in range(self.freq_channels):
            ax3.loglog(cls_mean[k], linestyle = "-", label = "f%s=%s" % (k+1, ("%.4f" % round(freqs[k], 4))))
        ax3.set_xlabel(r"$\ell$")
        ax3.set_xlim(xmin=0, xmax=10e3)
        ax3.set_ylim(ymax=9.e-10, ymin=1.e-15)
        #plt.legend(loc = 1 )
        ax3.set_title("{} Angular Power Spectra".format(self.name))
        #plt.show()
        fig3.savefig('/home/elimboto/Desktop/Forecast_Codes/hi_im_pipeline/cls.pdf', format='pdf', bbox_inches='tight')
        return cls_mean
    
    @property
    def plot_cls_separately(self):
        cls_mean = np.mean(self.ang_power_spectra, axis = 0)
        freqs = np.linspace(960, 1260, 30+1, endpoint=True)
        freqs= (freqs[1:] + freqs[:-1])/2.0
        
        for k in range(self.freq_channels):
            plt.figure()
            plt.loglog(cls_mean[k], linestyle = "-", label = "freq=%s" % ("%.4f" % round(freqs[k], 4)))
            plt.legend(loc = 1 )
            plt.ylim(ymax=9.e-10, ymin=1.e-15)
            plt.xlim(xmax=10e3, xmin=0)
            plt.legend(loc = 1 )
            if k == 0:
                    plt.title('{} Angular Power Spectra'.format(self.name))
            #plt.title("freq = {}".format("%.4f" % round(freq[i][0], 4)))
            plt.savefig('/home/elimboto/Desktop/Forecast_Codes/hi_im_pipeline/cls%s.pdf' % (k+1), bbox_inches='tight')
            #plt.show()
        return cls_mean

maps = sorted(glob.glob('/media/elimboto/21EE-093A/Test_maps/21cm_map*.h5'))
BINGO_21cm_signal = Sky_Maps(maps, 30, 768, 0, 0, 256, 130, 140, "BINGO")
BINGO_21cm_signal.plot_cls
BINGO_21cm_signal.plot_cls_separately
#BINGO_21cm_signal.ang_power_spectra
 





