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

* Rotation operator, just use complex notation

"""


import numpy as np
from lenspyx.utils_hp import almxfl, Alm
import healpy as hp

from delensalot.core.secondaries.operators import Operators, Operator, randomize, fg_phases


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
        mmax_out = lmax_out if mmax_out is None else mmax_out
        if not derivative:
            if self.disable:
                print("Lensing is disabled")
                eblm = np.atleast_2d(eblm)
                return eblm if out_sht_mode == "STANDARD" else eblm[0]
            
            #assert eblm.ndim == 1
            #elm2d = eblm.reshape((1, elm.size))
            result = self.ffi.lensgclm(np.atleast_2d(eblm), lmax_in, spin, lmax_out, mmax_out, gclm_out = gclm_out, backwards = backwards, out_sht_mode = out_sht_mode)
            if out_real:
                 result = q_pbgeom.synthesis(result.astype(np.complex128).copy(), 2, lmax_out, mmax_out, self.sht_tr)
            return result
        else:
            assert spin in [1, 3], spin
            return self.derivative(elm_wf = eblm, spin = spin, mmax_sol = lmax_out, q_pbgeom = q_pbgeom, out_real = out_real)

    def derivative(self, elm_wf, spin, mmax_sol, q_pbgeom, out_real = False):
        #note elm_wf is not always a emode only!!

        lmax = Alm.getlmax(elm_wf.size, mmax_sol) if elm_wf.ndim == 1 else Alm.getlmax(elm_wf[0].size, mmax_sol)
        i1, i2 = (2, -1) if spin == 1 else (-2, 3)
        fl = np.arange(i1, lmax + i1 + 1, dtype=float) * np.arange(i2, lmax + i2 + 1)
        fl[:spin] *= 0.
        fl = np.sqrt(fl)
        if elm_wf.ndim == 1:
            elm = almxfl(elm_wf, fl, mmax_sol, False)
        else:
            elm = np.array([almxfl(elm_wf[0], fl, mmax_sol, False), almxfl(elm_wf[1], fl, mmax_sol, False)])
        
        ffi = self.ffi.change_geom(q_pbgeom) if q_pbgeom is not self.ffi.pbgeom else self.ffi
        result = ffi.gclm2lenmap(elm, mmax_sol, spin, False)

        result = q_pbgeom.adjoint_synthesis(result, 2, mmax_sol, mmax_sol, self.sht_tr,
                                                apply_weights = True) if not out_real else result

        return result
    

    def get_qlms(self, filtr, eblm_dat: np.ndarray or list, elm_wf: np.ndarray, q_pbgeom: pbdGeometry, elm_wf_leg2:None or np.ndarray =None, which = "p", shift_1: int = 0, shift_2: int = 0, mean_field = False, filter_leg2 = None, cache = False):
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
        if elm_wf_leg2 is None:
            elm_wf_leg2 = elm_wf.copy()

        if filter_leg2 is None:
            filter_leg2 = filtr
        
        assert Alm.getlmax(eblm_dat[0].size, filtr.mmax_len) == filtr.lmax_len, (Alm.getlmax(eblm_dat[0].size, filtr.mmax_len), filtr.lmax_len)
        assert Alm.getlmax(eblm_dat[1].size, filtr.mmax_len) == filtr.lmax_len, (Alm.getlmax(eblm_dat[1].size, filtr.mmax_len), filtr.lmax_len)
        assert Alm.getlmax(elm_wf.size, filtr.mmax_sol) == filtr.lmax_sol, (Alm.getlmax(elm_wf.size, filtr.mmax_sol), filtr.lmax_sol)
        
        if cache:
            try:
                #resmap_r = self.cache["resmap_r"].copy()
                resmap_c = self.cache["resmap_c"].copy()
            except:
                resmap_c = np.empty((q_pbgeom.geom.npix(),), dtype=elm_wf.dtype)
                resmap_r = resmap_c.view(rtype[resmap_c.dtype]).reshape((resmap_c.size, 2)).T  # real view onto complex array
                filtr._get_irespmap(eblm_dat, elm_wf, q_pbgeom, map_out=resmap_r, which = which, mean_field = mean_field) # inplace onto resmap_c and resmap_r
                print("Chache lensing")
                self.cache = {}
                #self.cache["resmap_r"] = resmap_r.copy()
                self.cache["resmap_c"] = resmap_c.copy()
        else:
            resmap_c = np.empty((q_pbgeom.geom.npix(),), dtype=elm_wf.dtype)
            resmap_r = resmap_c.view(rtype[resmap_c.dtype]).reshape((resmap_c.size, 2)).T  # real view onto complex array
            filtr._get_irespmap(eblm_dat, elm_wf, q_pbgeom, map_out=resmap_r, which = which, mean_field = mean_field) # inplace onto resmap_c and resmap_r
    
        resmap_c = randomize(filtr, resmap_c, shift = shift_1)

        gcs_r = filter_leg2._get_gpmap(elm_wf_leg2, 3, q_pbgeom.geom, which = which)  # 2 pos.space maps, uses then complex view onto real array
        gcs_r = randomize(filtr, gcs_r, shift = shift_2)

        gc_c = resmap_c.conj() * gcs_r.T.view(ctype[gcs_r.dtype]).squeeze()  # (-2 , +3)
        #gc_c = resmap_c.conj() * np.ascontiguousarray(gcs_r.T).view(ctype[gcs_r.dtype]).squeeze()#NOTE!

        gcs_r = filter_leg2._get_gpmap(elm_wf_leg2, 1, q_pbgeom.geom, which = which)
        gcs_r = randomize(filtr, gcs_r, shift = shift_2)

        gc_c -= resmap_c * gcs_r.T.view(ctype[gcs_r.dtype]).squeeze().conj()  # (+2 , -1)
        #gc_c -= resmap_c * np.ascontiguousarray(gcs_r.T).view(ctype[gcs_r.dtype]).squeeze().conj()#NOTE!
        del resmap_c, resmap_r, gcs_r

        lmax_qlm, mmax_qlm = filtr.operators.lmax(which = which), filtr.operators.mmax(which = which)
        gc_r = gc_c.view(rtype[gc_c.dtype]).reshape((gc_c.size, 2)).T  # real view onto complex array
        gc = q_pbgeom.geom.adjoint_synthesis(gc_r, 1, lmax_qlm, mmax_qlm, filtr.operators.sht_tr)
        del gc_r, gc_c
        fl = - np.sqrt(np.arange(lmax_qlm + 1, dtype=float) * np.arange(1, lmax_qlm + 2))
        almxfl(gc[0], fl, mmax_qlm, True)
        almxfl(gc[1], fl, mmax_qlm, True)

        return gc


class Rotation(Operator):

    def __init__(self, name: str, lmax: int, mmax: int, sht_tr: int, disable: bool = False):
        super().__init__(name, lmax, mmax, sht_tr, disable)
        self.alpha = None

    def __call__(self, eblm, backwards = False, derivative = False, spin = None, q_pbgeom = None, 
                 lmax_in = None, lmax_out = None, mmax_out = None, out_sht_mode = "STANDARD", apply_weights = True, out_real = False):
        assert q_pbgeom is not None, "q_pbgeom cannot be None"

        #number of dimensions
        shape = eblm.shape


        if eblm.ndim == 1:
            assert hp.Alm.getlmax(eblm.size) == lmax_in, f"eblm and lmax_in mismatch {(hp.Alm.getlmax(eblm.size), lmax_in)}"
            eblm = np.array([eblm, np.zeros_like(eblm)])
        elif shape[0] == 1:
            assert hp.Alm.getlmax(eblm[0].size) == lmax_in, f"eblm and lmax_in mismatch {(hp.Alm.getlmax(eblm.size), lmax_in)}"
            eblm = np.array([eblm[0], np.zeros_like(eblm[0])])
        elif shape[0] == 2:
            assert hp.Alm.getlmax(eblm[0].size) == lmax_in, f"eblm and lmax_in mismatch {(hp.Alm.getlmax(eblm[0].size), lmax_in)}"


        if derivative:
            QU = q_pbgeom.synthesis(eblm.astype(np.complex128).copy(), 2, lmax_in, lmax_in, self.sht_tr)
            qumap = self.derivative_rotate_polarization(QU)
        else:
            if self.disable:
                print("Alpha is disabled")
                if not out_real:
                    return eblm if out_sht_mode == "STANDARD" else eblm[0]
                else:
                    return np.asfortranarray(q_pbgeom.synthesis(eblm.astype(np.complex128).copy(), 2, lmax_out, mmax_out, self.sht_tr))
                
            QU = q_pbgeom.synthesis(eblm.astype(np.complex128).copy(), 2, lmax_in, lmax_in, self.sht_tr)
            qumap =  self.rotate_polarization(QU, backwards)

        result = q_pbgeom.adjoint_synthesis(np.array(qumap), 2, lmax_out, mmax_out, self.sht_tr,
                                                apply_weights = apply_weights) if not out_real else np.asfortranarray(qumap)

        return result if out_sht_mode == "STANDARD" else result[0]


    def _set_alpha(self, alpha):
        if self.disable:
            alpha = np.zeros_like(alpha)
        self.alpha = alpha
        self.rotation = np.exp(1j * 2* alpha)
        self.inv_rotation = np.exp(-1j * 2* alpha)
        self.rotation_derivative = 2j * np.exp(1j * 2* alpha)

    @staticmethod
    def __rotate_polarization(QU, rotation):
        QpmU = QU[0] + 1j * QU[1]
        result = rotation*QpmU
        return np.array([result.real, result.imag])

    

    def set_alpha(self, alpha):
        if self.disable:
            alpha = np.zeros_like(alpha)
        self.alpha = alpha
        self.c, self.s = np.cos(2*alpha), np.sin(2*alpha)
        c, s = self.c, self.s
        self.rotation = np.array([[c, -s], [s, c]]) #R
        self.inv_rotation = np.array([[c, s], [-s, c]]) #R^\dagger

        self.rotation_derivative = np.array([[-2*s, 2*c], [-2*c, -2*s]]) #R^dagger derivative

    def set_field(self, field):
        self.set_alpha(field)

    @property
    def field(self):
        return self.alpha

    def rotate_polarization(self, QU, backwards=False):
        if self.alpha is not None:
            return self._rotate_polarization(QU, self.rotation if not backwards else self.inv_rotation)
        else:
            return QU

    def derivative_rotate_polarization(self, QU):
        return self._rotate_polarization(QU, self.rotation_derivative)

    @staticmethod
    def _rotate_polarization(QU, rotation):
        result = np.einsum('abc, bc->ac', rotation, QU)
        return result
    

    def get_qlms(self, filtr, eblm_dat: np.ndarray or list, elm_wf: np.ndarray, q_pbgeom: pbdGeometry, elm_wf_leg2:None or np.ndarray =None, which = "a", shift_1: int = 0, shift_2: int = 0, mean_field = False, filter_leg2 = None, cache = False):
        """Get lensing generaliazed QE consistent with filter assumptions

            Args:
                eblm_dat: input polarization maps (geom must match that of the filter)
                elm_wf: Wiener-filtered CMB maps (alm arrays)
                alm_wf_leg2: Wiener-filtered CMB maps of gradient leg, if different from ivf leg (alm arrays)
                q_pbgeom: scarf pbounded-geometry of for the position-space mutliplication of the legs

            All implementation signs are super-weird but end result should be correct...

        """
        
        if elm_wf_leg2 is None:
            elm_wf_leg2 = elm_wf.copy()
        if filter_leg2 is None:
            filter_leg2 = filtr

        assert Alm.getlmax(eblm_dat[0].size, filtr.mmax_len) == filtr.lmax_len, (Alm.getlmax(eblm_dat[0].size, filtr.mmax_len), filtr.lmax_len)
        assert Alm.getlmax(eblm_dat[1].size, filtr.mmax_len) == filtr.lmax_len, (Alm.getlmax(eblm_dat[1].size, filtr.mmax_len), filtr.lmax_len)
        assert Alm.getlmax(elm_wf.size, filtr.mmax_sol) == filtr.lmax_sol, (Alm.getlmax(elm_wf.size, filtr.mmax_sol), filtr.lmax_sol)

        if cache:
            try:
                eblm_dat_map = self.cache["eblm_dat_map"].copy()
            except:
                eblm_dat_map = filtr._get_irespmap(eblm_dat, elm_wf, q_pbgeom, which = which, mean_field = mean_field)#, map_out=resmap_r) # inplace onto resmap_c and resmap_r
                self.cache = {}
                self.cache["eblm_dat_map"] = eblm_dat_map.copy()
        else:
            eblm_dat_map = filtr._get_irespmap(eblm_dat, elm_wf, q_pbgeom, which = which, mean_field = mean_field)#, map_out=resmap_r) # inplace onto resmap_c and resmap_r

        eblm_dat_map = randomize(filtr, eblm_dat_map, shift = shift_1)
        
        if cache:
            try:
                eblm_wf_map = self.cache["eblm_wf_map"].copy()
            except:
                eblm_wf_map = filter_leg2._get_gpmap(elm_wf_leg2, 2, q_pbgeom, which = which)
                self.cache["eblm_wf_map"] = eblm_wf_map.copy()
        else:
            eblm_wf_map = filter_leg2._get_gpmap(elm_wf_leg2, 2, q_pbgeom, which = which)

        eblm_wf_map = randomize(filter_leg2, eblm_wf_map, shift = shift_2)

        gc = eblm_dat_map[0] * eblm_wf_map[0] + eblm_dat_map[1] * eblm_wf_map[1]
        
        lmax_qlm, mmax_qlm = filtr.operators.lmax(which = which), filtr.operators.mmax(which = which)
        return (-2)*q_pbgeom.geom.adjoint_synthesis(gc, 0, lmax_qlm, mmax_qlm, filtr.operators.sht_tr).squeeze()


class PatchyTau(Operator):

    def __init__(self, name: str, lmax: int, mmax: int, sht_tr: int, disable: bool = False):
        super().__init__(name, lmax, mmax, sht_tr, disable)
        self.tau = None

    def __call__(self, eblm, backwards = False, derivative = False, spin = None, q_pbgeom = None, 
                 lmax_in = None, lmax_out = None, mmax_out = None, out_sht_mode = "STANDARD", apply_weights = True, out_real = False):
        
        assert q_pbgeom is not None, "q_pbgeom cannot be None"

        #number of dimensions
        shape = eblm.shape


        if eblm.ndim == 1:
            assert hp.Alm.getlmax(eblm.size) == lmax_in, f"eblm and lmax_in mismatch {(hp.Alm.getlmax(eblm.size), lmax_in)}"
            eblm = np.array([eblm, np.zeros_like(eblm)])
        elif shape[0] == 1:
            assert hp.Alm.getlmax(eblm[0].size) == lmax_in, f"eblm and lmax_in mismatch {(hp.Alm.getlmax(eblm.size), lmax_in)}"
            eblm = np.array([eblm[0], np.zeros_like(eblm[0])])
        elif shape[0] == 2:
            assert hp.Alm.getlmax(eblm[0].size) == lmax_in, f"eblm and lmax_in mismatch {(hp.Alm.getlmax(eblm[0].size), lmax_in)}"


        if derivative:
            QU = q_pbgeom.synthesis(eblm.astype(np.complex128).copy(), 2, lmax_in, lmax_in, self.sht_tr)
            qumap = self.derivative_patchy(QU)
        else:
            if self.disable:
                print("Patchy is disabled")
                return eblm if out_sht_mode == "STANDARD" else eblm[0]

            QU = q_pbgeom.synthesis(eblm.astype(np.complex128).copy(), 2, lmax_in, lmax_in, self.sht_tr)
            qumap =  self.patchy(QU, backwards)

        result = q_pbgeom.adjoint_synthesis(np.array(qumap), 2, lmax_out, mmax_out, self.sht_tr,
                                                apply_weights = apply_weights) if not out_real else qumap

        return result if out_sht_mode == "STANDARD" else result[0]

    def set_tau(self, tau):
        if self.disable:
            tau = np.zeros_like(tau)
        self.tau = tau
        self.exp_tau = np.exp(-tau)

    def set_field(self, field):
        self.set_tau(field)

    @property
    def field(self):
        return self.tau

    def patchy(self, QU, backwards=False):
        #patchy suppresses information
        if self.tau is not None:
            return self._multiply(QU, self.exp_tau if not backwards else self.exp_tau)
        else:
            return QU

    def derivative_patchy(self, QU):
        return self._multiply(QU, -self.exp_tau)
    
    @staticmethod
    def _multiply(a, b):
        return np.einsum('ab, b -> ab', a, b)
    

    def get_qlms(self, filtr, eblm_dat: np.ndarray or list, elm_wf: np.ndarray, q_pbgeom: pbdGeometry, alm_wf_leg2:None or np.ndarray =None, which = "f", shift_1: int = 0, shift_2: int = 0, mean_field = False, filter_leg2 = None, cache = False):
        """Get lensing generaliazed QE consistent with filter assumptions

            Args:
                eblm_dat: input polarization maps (geom must match that of the filter)
                elm_wf: Wiener-filtered CMB maps (alm arrays)
                alm_wf_leg2: Wiener-filtered CMB maps of gradient leg, if different from ivf leg (alm arrays)
                q_pbgeom: scarf pbounded-geometry of for the position-space mutliplication of the legs

            All implementation signs are super-weird but end result should be correct...

        """
        assert alm_wf_leg2 is None

        if filter_leg2 is None:
            filter_leg2 = filtr

        assert Alm.getlmax(eblm_dat[0].size, filtr.mmax_len) == filtr.lmax_len, (Alm.getlmax(eblm_dat[0].size, filtr.mmax_len), filtr.lmax_len)
        assert Alm.getlmax(eblm_dat[1].size, filtr.mmax_len) == filtr.lmax_len, (Alm.getlmax(eblm_dat[1].size, filtr.mmax_len), filtr.lmax_len)
        assert Alm.getlmax(elm_wf.size, filtr.mmax_sol) == filtr.lmax_sol, (Alm.getlmax(elm_wf.size, filtr.mmax_sol), filtr.lmax_sol)

        eblm_dat_map = filtr._get_irespmap(eblm_dat, elm_wf, q_pbgeom, which = which, shift = shift_1, mean_field = mean_field)

        eblm_wf_map = filter_leg2._get_gpmap(elm_wf, 2, q_pbgeom, which = which, shift = shift_2) 
        gc = eblm_dat_map[0] * eblm_wf_map[0] + eblm_dat_map[1] * eblm_wf_map[1]
        lmax_qlm, mmax_qlm = filtr.operators.lmax(which = which), filtr.operators.mmax(which = which)
        return 2*q_pbgeom.geom.adjoint_synthesis(gc, 0, lmax_qlm, mmax_qlm, filtr.operators.sht_tr).squeeze()