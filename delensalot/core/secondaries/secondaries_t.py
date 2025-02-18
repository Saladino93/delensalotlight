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

* To improve speed. IVF is calculated multiple times per secondary. Just calculate it once, and cache it somewhere. Better a different object/module to interface with. 

* The Wiener filter has parts that could be cached. For example, if I do ABCX, and I calculate CX, I can cache it for later use. Need to think about this for speed.

* Clean code

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

    def __call__(self, tlm, lmax_in, spin, lmax_out, mmax_out = None, gclm_out = None, backwards = False, 
                 out_sht_mode: str = 'STANDARD', derivative = False, q_pbgeom = None, out_real = False):
        mmax_out = lmax_out if mmax_out is None else mmax_out
        if not derivative:
            if self.disable:
                print("Lensing is disabled")
                return tlm
            
            result = self.ffi.lensgclm(tlm, lmax_in, 0, lmax_out, mmax_out)
            if out_real:
                 result = q_pbgeom.synthesis(result.astype(np.complex128).copy(), 0, lmax_out, mmax_out, self.sht_tr)
            return result
        else:
            assert spin in [0], spin
            return self.derivative(tlm_wf = tlm, spin = spin, mmax_sol = lmax_out, q_pbgeom = q_pbgeom, out_real = out_real)


    def derivative(self, tlm_wf, spin, mmax_sol, q_pbgeom, out_real = False):
        """
        This basically gives the leg acting on a Wiener filtered map.
        In this case, it outputs a lensed (from gclm2lenmap) gradient (from fl)
        """


        #note elm_wf is not always a emode only!!

        assert spin in [0], spin

        lmax = Alm.getlmax(tlm_wf.size, mmax_sol)
        i1, i2 = (0, 0)
        fl = np.arange(i1, lmax + i1 + 1, dtype=float) * np.arange(i2, lmax + i2 + 1)
        fl[:spin] *= 0. #spin should be zero
        fl = np.sqrt(fl)
        tlm = almxfl(tlm_wf, fl, mmax_sol, False)
        
        ffi = self.ffi.change_geom(q_pbgeom) if q_pbgeom is not self.ffi.pbgeom else self.ffi
        result = ffi.gclm2lenmap(tlm, mmax_sol, spin, False)

        result = q_pbgeom.adjoint_synthesis(result, 0, mmax_sol, mmax_sol, self.sht_tr,
                                                apply_weights = True) if not out_real else result
        return result
    

    def get_qlms(self, filtr, tlm_dat: np.ndarray or list, tlm_wf: np.ndarray, q_pbgeom: pbdGeometry, tlm_wf_leg2:None or np.ndarray =None, which = "p", shift_1: int = 0, shift_2: int = 0, mean_field = False, filter_leg2 = None, cache = False):
        """From delensalot.MAP filter:
        
            Get lensing generaliazed QE consistent with filter assumptions

            Args:
                eblm_dat: input polarization maps (geom must match that of the filter)
                elm_wf: Wiener-filtered CMB maps (alm arrays)
                alm_wf_leg2: Wiener-filtered CMB maps of gradient leg, if different from ivf leg (alm arrays)
                q_pbgeom: scarf pbounded-geometry of for the position-space mutliplication of the legs

            All implementation signs are super-weird but end result should be correct...

        """

        #assert alm_wf_leg2 is None
        if tlm_wf_leg2 is None:
            tlm_wf_leg2 = tlm_wf.copy()

        if filter_leg2 is None:
            filter_leg2 = filtr
        
        assert Alm.getlmax(tlm_wf.size, filtr.mmax_sol) == filtr.lmax_sol, (Alm.getlmax(tlm_wf.size, filtr.mmax_sol), filtr.lmax_sol)
        assert Alm.getlmax(tlm_wf_leg2.size, filter_leg2.mmax_sol) == filter_leg2.lmax_sol, (Alm.getlmax(tlm_wf_leg2.size, filter_leg2.mmax_sol), filter_leg2.lmax_sol)

        ivf = filtr._get_irestmap(tlm_dat, tlm_wf, q_pbgeom)
        gwf = filtr._get_gtmap(tlm_wf_leg2, q_pbgeom)

        ivf = randomize(filtr, ivf, shift = shift_1)
        gwf = randomize(filter_leg2, gwf, shift = shift_2)

        d1 = ivf * gwf

        lmax_qlm, mmax_qlm = filtr.operators.lmax(which = which), filtr.operators.mmax(which = which)

        #G, C = q_pbgeom.geom.adjoint_synthesis(d1, 1, lmax_qlm, mmax_qlm, filtr.operators.sht_tr)
        G, C = q_pbgeom.geom.map2alm_spin(d1, 1, self.ffi.lmax_dlm, self.ffi.mmax_dlm, self.ffi.sht_tr, (-1., 1.))
        
        del d1
        fl = - np.sqrt(np.arange(lmax_qlm + 1, dtype=float) * np.arange(1, lmax_qlm + 2))
        almxfl(G, fl, mmax_qlm, True)
        almxfl(C, fl, mmax_qlm, True)
        return G, C


class AmplitudeModulation(Operator):

    def __init__(self, name: str, lmax: int, mmax: int, sht_tr: int, disable: bool = False):
        super().__init__(name, lmax, mmax, sht_tr, disable)
        self.tau = None

    def change_geom(self, geom):
        return self.ffi.change_geom(geom)



class NoiseModulation(AmplitudeModulation):
    def _set_noise(self, noise):
        self.noise = noise
    
    def set_field(self, field):
        self._set_noise(field)

    @property
    def field(self):
        return self.noise
    
    def __call__(self, tlm, lmax_in, spin, lmax_out, mmax_out = None, gclm_out = None, backwards = False, 
                 out_sht_mode: str = 'STANDARD', derivative = False, q_pbgeom = None, out_real = False):
        pass
    
    def get_qlms(self, filtr, tlm_dat: np.ndarray or list, tlm_wf: np.ndarray, q_pbgeom: pbdGeometry, tlm_wf_leg2:None or np.ndarray =None, which = "p", shift_1: int = 0, shift_2: int = 0, mean_field = False, filter_leg2 = None, cache = False):
        """This is to estimate a noise anisotropy.
        """

        #assert alm_wf_leg2 is None
        if tlm_wf_leg2 is None:
            tlm_wf_leg2 = tlm_wf.copy()

        if filter_leg2 is None:
            filter_leg2 = filtr
        
        assert Alm.getlmax(tlm_wf.size, filtr.mmax_sol) == filtr.lmax_sol, (Alm.getlmax(tlm_wf.size, filtr.mmax_sol), filtr.lmax_sol)
        assert Alm.getlmax(tlm_wf_leg2.size, filter_leg2.mmax_sol) == filter_leg2.lmax_sol, (Alm.getlmax(tlm_wf_leg2.size, filter_leg2.mmax_sol), filter_leg2.lmax_sol)

        ivf = filtr._get_irestmap(tlm_dat, tlm_wf, q_pbgeom)

        if not np.allclose(tlm_wf_leg2, tlm_wf):
            ivf2 = filter_leg2._get_irestmap(tlm_dat, tlm_wf_leg2, q_pbgeom)
            ivf2 = randomize(filter_leg2, ivf2, shift = shift_2)
        else:
            ivf = randomize(filtr, ivf, shift = shift_1)
            ivf2 = ivf

        d1 = ivf*ivf2

        lmax_qlm, mmax_qlm = filtr.operators.lmax(which = which), filtr.operators.mmax(which = which)
        #G, C = q_pbgeom.geom.adjoint_synthesis(d1, 1, lmax_qlm, mmax_qlm, filtr.operators.sht_tr)
        G = q_pbgeom.geom.map2alm(d1, self.ffi.lmax_dlm, self.ffi.mmax_dlm, self.ffi.sht_tr, (-1., 1.))
        
        return G


class Source(Operator):
    pass


class PatchyTau(AmplitudeModulation):

    def __init__(self, name: str, lmax: int, mmax: int, sht_tr: int, disable: bool = False):
        super().__init__(name, lmax, mmax, sht_tr, disable)
        self.tau = None