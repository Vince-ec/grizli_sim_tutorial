__author__ = 'Vince.ec'

import numpy as np
from grizli import model as griz_model
from scipy.interpolate import interp1d
import pysynphot as S
from astropy.table import Table

def Scale_model(data, sigma, model):
    return np.sum(((data * model) / sigma ** 2)) / np.sum((model ** 2 / sigma ** 2))


def  Gauss_dist(x, mu, sigma):
    return (1. / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def Min_sig(wv,c,ew,line,max_fl):
    U = 0
    sig_guess = 1.
    dx = 1.

    while np.abs(U - ew) > 0.1:
        flam = Gauss_dist(wv, line, sig_guess)
        flam *= (max_fl / max(flam))
        U = np.trapz(flam / c,wv)

        if np.abs(U - ew) < 0.1:
            break
        if ew - U  > 0:
            sig_guess +=dx
        if ew - U < 0:
            sig_guess -= dx
            dx/=10
            sig_guess += dx
            
    return sig_guess

def Gen_spec(wv_range, lines, ews, maxs):
    spec = np.repeat(1E-18, len(wv_range))
    c = np.repeat(1E-18, len(wv_range))
    
    emmision = np.zeros([len(lines),len(wv_range)])

    for i in range(len(lines)):
        sig = Min_sig(wv_range, c, ews[i], lines[i], maxs[i])
        emmision[i] = Gauss_dist(wv_range,lines[i],sig)
        emmision[i] *= ( maxs[i]/ np.max(emmision[i]))
        
        spec += emmision[i]
    return wv_range, spec

class Gen_sim(object):
    def __init__(self, redshift, wavelength, flux, signal_to_noise, minwv = 8000, maxwv = 11300):
        import pysynphot as S
        self.redshift = redshift
        self.wv = wavelength
        self.wv_obs = self.wv * (1 + self.redshift)
        self.fl = flux
        self.SNR = signal_to_noise
        self.er = self.fl / self.SNR
        """ 
        self.flt_input - image flt which contains the object you're interested in modeling 
        **
        self.beam - information used to make models
        **
        self.redshift - redshift of simulation
        **
        self.wv - input wavelength array of spectrum
        **
        self.wv_obs - input wavelength array of spectrum
        **
        self.fl - input flux array
        **
        self.er - output error array 
        **
        self.SNR - signal to noise ratio used to generate error
        **
        self.mfl - forward modeled flux array
        **
        self.mfl - forward modeled flux array
        **        
        self.mwv - output wavelength array in the observed frame
        **
        self.flx_err - output flux array of spectra perturb by the errors

        """      
        
        ## set flt file, we're using a good example galaxy
        self.flt_input = 's47677_flt.fits' 


        ## Create Grizli model object
        sim_g102 = griz_model.GrismFLT(grism_file='', verbose=False,
                                         direct_file=self.flt_input,
                                         force_grism='G102')

        sim_g102.photutils_detection(detect_thresh=.025, verbose=True, save_detection=True)

        c = Table.read('s47677_flt.detect.cat',format='ascii')
        sim_g102.catalog = c
        keep = sim_g102.catalog['mag'] < 29

        sim_g102.compute_full_model(ids=c['id'][keep], mags=c['mag'][keep], verbose=False)

        ## Grab object near the center of the image
        dr = np.sqrt((sim_g102.catalog['x_flt'] - 579) ** 2 + (sim_g102.catalog['y_flt'] - 522) ** 2)
        ix = np.argmin(dr)
        id = sim_g102.catalog['id'][ix]

        ## Spectrum cutouts
        self.beam = griz_model.BeamCutout(sim_g102, beam=sim_g102.object_dispersers[id][2]['A'], conf=sim_g102.conf)

        ## create basis model for sim
        spec = S.ArraySpectrum(self.wv, self.fl, fluxunits='flam')
        spec = spec.redshift(self.redshift).renorm(1., 'flam', S.ObsBandpass('wfc3,ir,f105w'))
        spec.convert('flam')
        
        ## Compute the models
        self.beam.compute_model(spectrum_1d=[spec.wave, spec.flux])

        ## Extractions the model (error array here is meaningless)
        w, f, e = self.beam.beam.optimal_extract(self.beam.model, bin=0)

        ## Get sensitivity function
        sens_wv, sens_fl = [self.beam.beam.lam, self.beam.beam.sensitivity / np.max(self.beam.beam.sensitivity)]

        ## adjust model flux  
        IDX = [U for U in range(len(self.wv_obs)) if 8000 < self.wv_obs[U] < 11000]
        
        isens = interp1d(sens_wv, sens_fl)(self.wv_obs[IDX])

        ifl = interp1d(w, f)(self.wv_obs[IDX])

        C = Scale_model(self.fl[IDX], self.er[IDX], ifl / isens)
               
        ##trim and save outputs
        IDX = [U for U in range(len(w)) if minwv <= w[U] <= maxwv]

        self.mfl = C * f[IDX] / sens_fl[IDX]
        self.mwv = w[IDX]
        self.mer = np.mean(f[IDX]) / sens_fl[IDX]
        self.mer /= np.trapz(self.mer,self.mwv)
        self.mer *= np.trapz(self.mfl,self.mwv) / self.SNR
        
        
    def Sim_data(self):
        self.flx_err = np.abs(self.mfl + np.random.normal(0, self.mer))