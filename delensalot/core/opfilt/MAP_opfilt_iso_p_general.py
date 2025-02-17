"""Scarf-geometry based inverse-variance filters, inclusive of CMB lensing remapping

    This module collects filter instances working on idealized skies with homogeneous or colored noise spectra



"""
import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end
import numpy as np
from scipy.interpolate import UnivariateSpline as spl

from lenspyx import remapping
from lenspyx.utils_hp import almxfl, Alm, synalm, alm2cl
from lenspyx.utils import timer, cli
from lenspyx.remapping.utils_geom import pbdGeometry
from lenspyx.remapping.deflection_028 import rtype, ctype
from lenspyx.remapping import utils_geom

from delensalot.utils import clhash
from delensalot.core.opfilt import opfilt_base, MAP_opfilt_aniso_p
from delensalot.core.secondaries import secondaries

from plancklens import utils as putils


class dot_op:
    def __init__(self, lmax: int, mmax: int or None, lmin=0):
        """scalar product operation for cg inversion

            Args:
                lmax: maximum multipole defining the alm layout
                mmax: maximum m defining the alm layout (defaults to lmax if None or < 0)


        """
        if mmax is None or mmax < 0: mmax = lmax
        self.lmax = lmax
        self.mmax = min(mmax, lmax)
        self.lmin = int(lmin)

    def __call__(self, elm1, elm2):
        assert elm1.size == Alm.getsize(self.lmax, self.mmax), (elm1.size, Alm.getsize(self.lmax, self.mmax))
        assert elm2.size == Alm.getsize(self.lmax, self.mmax), (elm2.size, Alm.getsize(self.lmax, self.mmax))
        return np.sum(alm2cl(elm1, elm2, self.lmax, self.mmax, None)[self.lmin:] * (2 * np.arange(self.lmin, self.lmax + 1) + 1))


class fwd_op:
    """Forward operation for polarization-only, no primordial B power cg filter


    """
    def __init__(self, s_cls:dict, ninv_filt:MAP_opfilt_aniso_p.alm_filter_ninv_wl):
        self.iclee = cli(s_cls['ee'])
        self.ninv_filt = ninv_filt
        self.lmax_sol = ninv_filt.lmax_sol
        self.mmax_sol = ninv_filt.mmax_sol

    def hashdict(self):
        return {'iclee': clhash(self.iclee),
                'n_inv_filt': self.ninv_filt.hashdict()}

    def __call__(self, elm):
        return self.calc(elm)

    def calc(self, elm):
        nlm = np.copy(elm)
        nlm = self.ninv_filt.apply_alm(nlm)
        nlm += almxfl(elm, self.iclee, self.mmax_sol, False)
        almxfl(nlm, self.iclee > 0., self.mmax_sol, True)
        return nlm


pre_op_dense = None # not implemented
apply_fini = MAP_opfilt_aniso_p.apply_fini

def _extend_cl(cl, lmax):
    """Forces input to an array of size lmax + 1

    """
    if np.isscalar(cl):
        return np.ones(lmax + 1, dtype=float) * cl
    ret = np.zeros(lmax + 1, dtype=float)
    ret[:min(len(cl), lmax+1)]= np.copy(cl[:min(len(cl), lmax+1)])
    return ret


class alm_filter_nlev_wl(opfilt_base.alm_filter_wl):
    def __init__(self, ninv_geom:utils_geom.Geom, nlev_p:float or np.ndarray, transf:np.ndarray, unlalm_info:tuple, lenalm_info:tuple,
                 transf_b:None or np.ndarray=None, nlev_b:None or float or np.ndarray=None, wee=True, verbose=False,
                 operators = secondaries.Operators):
        r"""Version of alm_filter_ninv_wl for full-sky maps filtered with homogeneous noise levels


                Args:
                    nlev_p: CMB-E filtering noise level in uK-amin
                            (to input colored noise cls, can feed in an array. Size must match that of the transfer fct)
                    ffi: lenscarf deflection instance
                    transf: CMB E-mode transfer function (beam, pixel window, mutlipole cuts, ...)
                    unlalm_info: lmax and mmax of unlensed CMB
                    lenalm_info: lmax and mmax of lensed CMB (greater or equal the transfer lmax)
                    transf_b(optional): CMB B-mode transfer function (if different from E)
                    nlev_b(optional): CMB-B filtering noise level in uK-amin
                             (to input colored noise cls, can feed in an array. Size must match that of the transfer fct)
                    wee: includes EE-like term in generalized QE if set

                Note:
                    All operations are in harmonic space.
                    Mode exclusions can be implemented setting the transfer fct to zero
                    (but the instance still expects the Elm and Blm arrays to have the same formal lmax)


        """
        lmax_sol, mmax_sol = unlalm_info
        lmax_len, mmax_len = lenalm_info
        lmax_transf = max(len(transf), len(transf if transf_b is None else transf_b)) - 1
        nlev_e = nlev_p
        nlev_b = nlev_p if nlev_b is None else nlev_b

        super().__init__(lmax_sol, mmax_sol, None)
        self.lmax_len = min(lmax_len, lmax_transf)
        self.mmax_len = min(mmax_len, self.lmax_len)

        transf_elm = transf
        transf_blm = transf_b if transf_b is not None else transf

        nlev_elm = _extend_cl(nlev_e, lmax_len)
        nlev_blm = _extend_cl(nlev_b, lmax_len)

        self.inoise_2_elm  = _extend_cl(transf_elm ** 2, lmax_len) * cli(nlev_elm ** 2) * (180 * 60 / np.pi) ** 2
        self.inoise_1_elm  = _extend_cl(transf_elm ** 1 ,lmax_len) * cli(nlev_elm ** 2) * (180 * 60 / np.pi) ** 2

        self.inoise_2_blm = _extend_cl(transf_blm ** 2, lmax_len) * cli(nlev_blm ** 2) * (180 * 60 / np.pi) ** 2
        self.inoise_1_blm = _extend_cl(transf_blm ** 1, lmax_len) * cli(nlev_blm ** 2) * (180 * 60 / np.pi) ** 2

        self.transf_elm  = _extend_cl(transf_elm, lmax_len)
        self.transf_blm  = _extend_cl(transf_blm, lmax_len)

        self.nlev_elm = nlev_elm
        self.nlev_blm = nlev_blm

        self.verbose = verbose
        self.wee = wee
        self.tim = timer(True, prefix='opfilt')

        self.operators = operators
        self.ninv_geom = ninv_geom

    def get_febl(self):
        return np.copy(self.inoise_2_elm), np.copy(self.inoise_2_blm)

    def set_field(self, ffi:remapping.deflection, which = "p"):
        #self.operators.set_field(ffi, which = which)
        pass

    def dot_op(self):
        return dot_op(self.lmax_sol, self.mmax_sol)

    def apply_alm(self, elm:np.ndarray):
        """Applies operator Y^T N^{-1} Y (now  bl ** 2 / n, where D is lensing, bl the transfer function)

        """
        # Forward lensing here
        self.tim.reset()
        lmax_unl = Alm.getlmax(elm.size, self.mmax_sol)
        assert lmax_unl == self.lmax_sol, (lmax_unl, self.lmax_sol)

        elm_2d = elm.reshape((1, elm.size))
        eblm = self.operators(elm_2d, lmax_in = self.mmax_sol, spin = 2, lmax_out = self.lmax_len, mmax_out = self.lmax_len,
                              backwards=False, q_pbgeom = self.ninv_geom)
        self.tim.add('lensgclm fwd')
        almxfl(eblm[0], self.inoise_2_elm, self.mmax_len, inplace=True)
        almxfl(eblm[1], self.inoise_2_blm, self.mmax_len, inplace=True)
        self.tim.add('transf')
        # NB: inplace is fine but only if precision of elm array matches that of the interpolator
        eblm = self.operators(eblm, lmax_in = self.mmax_len, spin = 2, lmax_out = self.lmax_sol, mmax_out = self.mmax_sol,
                                 backwards=True, gclm_out=elm_2d, out_sht_mode='GRAD_ONLY', q_pbgeom = self.ninv_geom, apply_weights = True)
        #NOTE: elm_2d, for lensing and rotation, how does this work?

        #raise NotImplementedError
        
        self.tim.add('lensgclm bwd')
        if self.verbose:
            print(self.tim)

        return eblm

    def apply_map(self, eblm:np.ndarray):
        """Applies noise operator in place"""
        almxfl(eblm[0], self.inoise_1_elm * cli(self.transf_elm), self.mmax_len, True)
        almxfl(eblm[1], self.inoise_1_blm * cli(self.transf_elm), self.mmax_len, True)


    def get_qlms(self, eblm_dat: np.ndarray or list, elm_wf: np.ndarray, q_pbgeom: pbdGeometry, alm_wf_leg2:None or np.ndarray =None, which = "p", shift_1: int = 0, shift_2: int = 0, mean_field = False, filter_leg2 = None, cache = False):
        """
        If you want to get a disconnected part, you do this: 
        0.5*[qlms(shift_A, shift_B)+qlms(shift_B, shift_A)]
        """
        assert which in ["p", "a", "f"], print("Operator must be one of 'p', 'a', 'f' ")

        return self.operators.get(which = which).get_qlms(self, eblm_dat, elm_wf, q_pbgeom, alm_wf_leg2, which = which, shift_1 = shift_1, shift_2 = shift_2, mean_field = mean_field, filter_leg2 = filter_leg2, cache = cache)

    def _p2h(self, h, lmax):
        if h == 'p':
            return np.ones(lmax + 1, dtype=float)
        elif h == 'k':
            return 0.5 * np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2, dtype=float)
        elif h == 'd':
            return np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2), dtype=float)
        else:
            assert 0, h + ' not implemented'

    def _h2p(self, h, lmax): return cli(self._p2h(h, lmax))


    def synalm(self, unlcmb_cls:dict, cmb_phas=None, seed = 0, get_unlelm=True):
        """Generate some dat maps consistent with noise filter fiducial ingredients

            Note:
                Feeding in directly the unlensed CMB phase can be useful for paired simulations.
                In this case the shape must match that of the filter unlensed alm array

        """
        elm = synalm(unlcmb_cls['ee'], self.lmax_sol, self.mmax_sol, seed = seed) if cmb_phas is None else cmb_phas
        assert Alm.getlmax(elm.size, self.mmax_sol) == self.lmax_sol, (Alm.getlmax(elm.size, self.mmax_sol), self.lmax_sol)
        
        eblm = self.operators(np.atleast_2d(elm), lmax_in = self.mmax_sol, spin = 2, lmax_out = self.lmax_len, mmax_out = self.lmax_len,
                              backwards=False, q_pbgeom = self.ninv_geom)
        
        almxfl(eblm[0], self.transf_elm, self.mmax_len, True)
        almxfl(eblm[1], self.transf_blm, self.mmax_len, True)

        eblm[0] += synalm((np.ones(self.lmax_len + 1) * (self.nlev_elm / 180 / 60 * np.pi) ** 2) * (self.transf_elm > 0), self.lmax_len, self.mmax_len, seed = seed+1000)
        eblm[1] += synalm((np.ones(self.lmax_len + 1) * (self.nlev_blm / 180 / 60 * np.pi) ** 2) * (self.transf_blm > 0), self.lmax_len, self.mmax_len, seed = seed+1000)
        return np.array(eblm)

    def get_qlms_mf(self, h, mfkey, q_pbgeom:pbdGeometry, mchain, lmax_qlm, mmax_qlm, phas=None, cls_filt:dict or None=None):
        """Mean-field estimate using tricks of Carron Lewis appendix


        """
        if mfkey in [1]: # This should be B^t x, D dC D^t B^t Covi x, x random phases in alm space
            if phas is None:
                phas = np.array([synalm(np.ones(self.lmax_len + 1, dtype=float), self.lmax_len, self.mmax_len),
                                 synalm(np.ones(self.lmax_len + 1, dtype=float), self.lmax_len, self.mmax_len)])
            assert Alm.getlmax(phas[0].size, self.mmax_len) == self.lmax_len
            assert Alm.getlmax(phas[1].size, self.mmax_len) == self.lmax_len

            soltn = np.zeros(Alm.getsize(self.lmax_sol, self.mmax_sol), dtype=complex)
            mchain.solve(soltn, phas, dot_op=self.dot_op())

            almxfl(phas[0], 0.5 * self.transf_elm, self.mmax_len, True)
            almxfl(phas[1], 0.5 * self.transf_blm, self.mmax_len, True)

            #phas is basically IVF leg
            #soltn is the WF leg
            G_total = []

            mean_field = True

            for o in self.operators:
                which = o.name
                if which == 'p':

                    G, C = self.get_qlms(phas, soltn, q_pbgeom, which = which, mean_field = mean_field)
                    almxfl(G, self._h2p(h, lmax_qlm), mmax_qlm, True)
                    almxfl(C, self._h2p(h, lmax_qlm), mmax_qlm, True)
                    G_total.append(G)

                    if "o" in self.operators.names:
                        G_total.append(C)

                elif which == 'o':
                    if "p" not in self.operators.names:
                        G, C = self.get_qlms(phas, soltn, q_pbgeom, which = which, mean_field = mean_field)
                        almxfl(C, self._h2p(h, lmax_qlm), mmax_qlm, True)
                    else:
                        print("Skip gradient o, already added.")
                else:
                    G = self.get_qlms(phas, soltn, q_pbgeom, which = which, mean_field = mean_field)
                    G_total.append(G)

            #print("LEN GTOTAL", len(G_total))

            #repmap, impmap = q_pbgeom.geom.synthesis(phas, 2, self.lmax_len, self.mmax_len, self.operators.sht_tr)

            #Gs, Cs = self._get_gpmap(soltn, 3, q_pbgeom)  # 2 pos.space maps
            #GC = (repmap - 1j * impmap) * (Gs + 1j * Cs)  # (-2 , +3)
            #Gs, Cs = self._get_gpmap(soltn, 1, q_pbgeom)
            #GC -= (repmap + 1j * impmap) * (Gs - 1j * Cs)  # (+2 , -1)
            #del repmap, impmap, Gs, Cs
        elif mfkey in [0]: # standard gQE, quite inefficient but simple
            assert phas is None, 'discarding this phase anyways'
            elm_pha, eblm_dat = self.synalm(cls_filt)
            eblm_dat = np.array(eblm_dat)
            elm_wf = np.zeros(Alm.getsize(self.lmax_sol, self.mmax_sol), dtype=complex)
            mchain.solve(elm_wf, eblm_dat, dot_op=self.dot_op())
            return self.get_qlms(eblm_dat, elm_wf, q_pbgeom)

        else:
            assert 0, mfkey + ' not implemented'

        lmax_qlm = self.operators.lmax
        mmax_qlm = self.operators.mmax

        return np.array(G_total)
    

    def _get_irespmap(self, eblm_dat:np.ndarray, eblm_wf:np.ndarray, q_pbgeom:pbdGeometry, map_out=None, which = "p", shift: int = 0, mean_field = False):
        """Builds inverse variance weighted map to feed into the QE


            :math:`B^t N^{-1}(X^{\rm dat} - B D X^{WF})`


        """
        assert len(eblm_dat) == 2

        if mean_field:
            return q_pbgeom.geom.synthesis(eblm_dat, 2, self.lmax_len, self.mmax_len, self.operators.sht_tr, map=map_out)


        eblm_wf_r = eblm_wf.copy()
        eblm_dat_r = eblm_dat.copy()

        ebwf = self.operators(np.atleast_2d(eblm_wf_r), lmax_in = self.mmax_sol, spin = 2, lmax_out = self.lmax_len, mmax_out = self.mmax_len, q_pbgeom = self.ninv_geom, ignore = ["o"] if which == "p" else [])

        almxfl(ebwf[0], (-1) * self.transf_elm, self.mmax_len, True)
        almxfl(ebwf[1], (-1) * self.transf_blm, self.mmax_len, True)
        ebwf += eblm_dat_r
        almxfl(ebwf[0], self.inoise_1_elm * 0.5 * self.wee, self.mmax_len, True)  # Factor of 1/2 because of \dagger rather than ^{-1}
        almxfl(ebwf[1], self.inoise_1_blm * 0.5,            self.mmax_len, True)

        return q_pbgeom.geom.synthesis(ebwf, 2, self.lmax_len, self.mmax_len, self.operators.sht_tr, map=map_out)

    def _get_gpmap(self, elm_wf:np.ndarray, spin:int, q_pbgeom:pbdGeometry, which = "p", shift: int = 0):
        """Wiener-filtered gradient leg to feed into the QE
        """
        assert elm_wf.ndim == 1
        assert Alm.getlmax(elm_wf.size, self.mmax_sol) == self.lmax_sol
        lmax = Alm.getlmax(elm_wf.size, self.mmax_sol)

        result = self.operators(eblm = elm_wf, backwards = False, lmax_in = lmax, spin = spin, lmax_out = self.mmax_sol, q_pbgeom = self.ninv_geom if which not in ["p", "o"] else q_pbgeom, which = which, ignore = ["o"] if which == "p" else []) #NOTE: CHECK IGNORE
        return result
        
class pre_op_diag:
    """Cg-inversion diagonal preconditioner


    """
    def __init__(self, s_cls:dict, ninv_filt:alm_filter_nlev_wl):
        assert len(s_cls['ee']) > ninv_filt.lmax_sol, (ninv_filt.lmax_sol, len(s_cls['ee']))
        lmax_sol = ninv_filt.lmax_sol
        ninv_fel, ninv_fbl = ninv_filt.get_febl()
        if len(ninv_fel) - 1 < lmax_sol: # We extend the transfer fct to avoid predcon. with zero (~ Gauss beam)
            log.debug("PRE_OP_DIAG: extending transfer fct from lmax %s to lmax %s"%(len(ninv_fel)-1, lmax_sol))
            assert np.all(ninv_fel >= 0)
            nz = np.where(ninv_fel > 0)
            spl_sq = spl(np.arange(len(ninv_fel), dtype=float)[nz], np.log(ninv_fel[nz]), k=2, ext='extrapolate')
            ninv_fel = np.exp(spl_sq(np.arange(lmax_sol + 1, dtype=float)))
        flmat = cli(s_cls['ee'][:lmax_sol + 1]) + ninv_fel[:lmax_sol + 1]
        self.flmat = cli(flmat) * (s_cls['ee'][:lmax_sol + 1] > 0.)
        self.lmax = ninv_filt.lmax_sol
        self.mmax = ninv_filt.mmax_sol

    def __call__(self, eblm):
        return self.calc(eblm)

    def calc(self, elm):
        assert Alm.getsize(self.lmax, self.mmax) == elm.size, (self.lmax, self.mmax, Alm.getlmax(elm.size, self.mmax))
        return almxfl(elm, self.flmat, self.mmax, False)


def calc_prep(eblm:np.ndarray, s_cls:dict, ninv_filt:alm_filter_nlev_wl):
    """cg-inversion pre-operation

        This performs :math:`D_\phi^t B^t N^{-1} X^{\rm dat}`

        Args:
            eblm: input data polarisation elm and blm
            s_cls: CMB spectra dictionary (here only 'ee' key required)
            ninv_filt: inverse-variance filtering instance


    """
    assert isinstance(eblm, np.ndarray) and eblm.ndim == 2
    assert Alm.getlmax(eblm[0].size, ninv_filt.mmax_len) == ninv_filt.lmax_len, (Alm.getlmax(eblm[0].size, ninv_filt.mmax_len), ninv_filt.lmax_len)
    eblmc = np.empty_like(eblm)

    eblmc[0] = almxfl(eblm[0], ninv_filt.inoise_1_elm, ninv_filt.mmax_len, False)
    eblmc[1] = almxfl(eblm[1], ninv_filt.inoise_1_blm, ninv_filt.mmax_len, False)

    elm = ninv_filt.operators(eblmc, lmax_in = ninv_filt.mmax_len, spin = 2, lmax_out = ninv_filt.lmax_sol, mmax_out = ninv_filt.mmax_sol,
                                      backwards = True, out_sht_mode='GRAD_ONLY', q_pbgeom = ninv_filt.ninv_geom, apply_weights = True).squeeze()
    almxfl(elm, s_cls['ee'] > 0., ninv_filt.mmax_sol, True)

    return elm