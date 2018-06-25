from sim_tools import Gen_spec, Gen_sim
import numpy as np

wv,fl = Gen_spec(np.arange(4000,9000,0.1),[6549,6564,6586],[3,10,2],[2E-18,1E-17,3E-18])
test = Gen_sim(redshift=0.5,wavelength=wv,flux=fl,signal_to_noise=20)
test.Sim_data()

np.save('test',[test.mwv,test.mfl,test.flx_err,test.mer])