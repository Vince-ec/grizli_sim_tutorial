import grizli
import numpy as np
from scipy.interpolate import interp1d
from astropy.table import Table
import astropy.wcs as pywcs
import pysynphot as S

def Mag(band):
    magnitude=25-2.5*np.log10(band)
    return magnitude

class Lyalpha_sim(object):
    
    def __init__(self, mag, EW, rshift, flt_input, mosaic, segment_map, reference_catalog, galaxy_id, 
                 pad = 100, wv_range = np.arange(6000,13000,10)):
        self.wv_range = wv_range
        self.mag = mag
        self.EW = EW
        self.rshift = rshift
        self.flt_input = flt_input
        self.mosaic = mosaic
        self.segment_map = segment_map
        self.reference_catalog = reference_catalog
        self.galaxy_id = galaxy_id
        self.pad = pad
        
        self.wave_1D = np.nan
        self.flux_1D = np.nan
        
        """ 
        self.mag - input magnitude of galaxy
        **
        self.EW - model equivalent width
        **
        self.rshift - model redshift 
        **
        self.flt_input - grism flt (not image flt) which contains the object you're interested in modeling, this
                         will tell Grizli the PA
        **                 
        self.mosaic - mosaic image 
        **
        self.segment_map - segmentation map for the corresponding mosaic
        **
        self.reference_catalog - reference catalog which matches the segmentation map, currently this must be an 
                                 ascii file, this can be changed later to include fits files
        **
        self.galaxy_id - ID of the object you want to model
        **
        self.pad - Grizli uses this to add extra pixels to the edge of an image to account for galaxies near the 
                   edge, 100 is usually enough
        **
        self.wv_range - wavelength of the redshifted spectra, the default is recommended as it covers the entire 
                        G102 window at a good resolution.
        **
        self.lya_dist - output model spectra
        **
        self.sim_2D - output 2D simulation
        **
        self.cutout - cutout of galaxy
        **
        self.wave_1D - output extracted 1D wavelength array, this will only populate when you run Extract_1D 
        **
        self.flux_1D - output extracted 1D flux (F_lam) array, this will only populate when you run Extract_1D 
        """        
        
        ## Simulate Ly_alpha spectra
        lya = 1216. * (1 + self.rshift)
        f0 =  3.631E-20
        flam = (1 + self.rshift ) * f0 * (2.998E18 / self.wv_range**2) * 10**(-0.4 * self.mag)
        sigma_lambda = 3. 
        fgauss = np.exp( - ((self.wv_range - lya) / sigma_lambda)**2 / 2.0)

        self.lya_dist = self.EW * flam / np.sqrt(2.0*np.pi) / sigma_lambda * fgauss
    
        ## Create Grizli model object
        sim_g102 = grizli.model.GrismFLT(grism_file = self.flt_input, verbose = False,
                                         ref_file = self.mosaic, seg_file = self.segment_map,
                                         force_grism = 'G102', pad = self.pad)

        ref_cat=Table.read(self.reference_catalog, format='ascii')

        sim_cat=sim_g102.blot_catalog(ref_cat, sextractor=False)

        sim_g102.compute_full_model(ids=sim_cat['id'], mags=Mag(sim_cat['f_F125W']))

        obj_mag = Mag(ref_cat['f_F125W'][np.argwhere(ref_cat['id']==self.galaxy_id)])

        ## normalize and redshift spectra
        spec = S.ArraySpectrum(self.wv_range,self.lya_dist, fluxunits='flam')
        spec = spec.redshift(0).renorm(1., 'flam', S.ObsBandpass('wfc3,ir,f105w'))
        spec.convert('flam')

        ## Compute the models
        beam_g102 = grizli.model.BeamCutout(sim_g102, sim_g102.object_dispersers[self.galaxy_id]['A'])
        
        beam_g102.compute_model(id = self.galaxy_id, spectrum_1d=[spec.wave,spec.flux], scale = np.max(self.lya_dist))

        self.sim_2D = beam_g102.model

        self.cutout = beam_g102
        
    def Extract_1D(self):
        
        ## Extract the model
        w, f, e = self.cutout.beam.optimal_extract(self.cutout.model,bin=0)
        
        ## Get sensitivity function
        fwv, ffl = [self.cutout.beam.lam, self.cutout.beam.sensitivity / np.max(self.cutout.beam.sensitivity)]
        filt = interp1d(fwv, ffl)

        ## Clip model to fit in interpolation, then remove sensitivity function
        clip = []
        for i in range(len(w)):
            if fwv[0] < w[i] < fwv[-1]:
                clip.append(i)

        w = w[clip]
        f = f[clip]

        f /= filt(w)

        ID1=[U for U in range(len(w)) if 7900 < w[U] <11300]
        
        self.wave_1D = w[ID1]
        self.flux_1D = f[ID1]