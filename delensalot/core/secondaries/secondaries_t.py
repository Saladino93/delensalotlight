"""
Quick and dirty code implementing secondaries for the polarization only case. Still work in progress.

Long-term some parts will be moved out.

TODO:

* Operators class,implement chain of operators in such a way that I get the input and outputs compatible with each other
  #e.g. f1 input is real output is complex, f2 input is complex output is complex, f3 input is real output is real
  #perhaps some decorators that are able to convert outputs when necessary
  #also, need to build some graph at the start of the program
  #or just make all the functions in the same format. then it is even simpler
  #i will just pay the price at the end, but I think I am actually gaining here
  
* Clean code

* Rotation operator, just use complex notation

"""
from delensalot.core.secondaries.operators import Operator, randomize, fg_phases

import numpy as np
from lenspyx.utils_hp import almxfl, Alm
import healpy as hp

from lenspyx.remapping.deflection_028 import rtype, ctype
from lenspyx.remapping.utils_geom import pbdGeometry


class Lensing(Operator):

    def _set_ffi(self, ffi):
        self.ffi = ffi

    def change_geom(self, geom):
        return self.ffi.change_geom(geom)
    
    def set_field(self, field):
        self._set_ffi(field)

    @property
    def field(self):
        return self.ffi

    def __call__(self, eblm, lmax_in, spin, lmax_out, mmax_out = None, gclm_out = None, backwards = False, 
                 out_sht_mode: str = 'STANDARD', derivative = False, q_pbgeom = None, out_real = False):
        pass

    def derivative(self, elm_wf, spin, mmax_sol, q_pbgeom, out_real = False):
        pass

    def get_qlms(self, filtr, eblm_dat: np.ndarray or list, elm_wf: np.ndarray, q_pbgeom: pbdGeometry, elm_wf_leg2:None or np.ndarray =None, which = "p", shift_1: int = 0, shift_2: int = 0, mean_field = False, filter_leg2 = None, cache = False):
        pass



class AmplitudeModulation(Operator):

    def __init__(self, name: str, lmax: int, mmax: int, sht_tr: int, disable: bool = False):
        super().__init__(name, lmax, mmax, sht_tr, disable)
        self.tau = None



class NoiseModulation(AmplitudeModulation):
    pass


class Source(Operator):
    pass


class PatchyTau(AmplitudeModulation):

    def __init__(self, name: str, lmax: int, mmax: int, sht_tr: int, disable: bool = False):
        super().__init__(name, lmax, mmax, sht_tr, disable)
        self.tau = None