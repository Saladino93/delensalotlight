"""Module for curved-sky iterative lensing estimation

    Version revised on March 2023

        Among the changes:
            * delensalot'ed this with great improvements in execution time
            * novel and more stable way of calculating the delfection angles and inverses
            * optionally change main variable from plm to klm or dlm with expected better behavior ?
            * rid of alm2rlm which was just wasting a little bit of time and loads of memory
            * abstracted bfgs with cacher and dot_op



    #FIXME: loading of total gradient seems mixed up with loading of quadratic gradient...
    #TODO: make plm0 possibly a path?
    #FIXME: Chh = 0 not resulting in 0 estimate
"""

import os
from os.path import join as opj
import shutil
import time
import sys
import numpy as np

import copy

import logging
log = logging.getLogger(__name__)
from logdecorator import log_on_start, log_on_end

from plancklens.qcinv import multigrid

import lenspyx.remapping.utils_geom as utils_geom
from lenspyx.remapping.utils_geom import pbdGeometry, pbounds

from delensalot.utils import cli, read_map
from delensalot.utility.utils_hp import Alm, almxfl, alm2cl
from delensalot.utility import utils_qe

from delensalot.core import cachers
from delensalot.core.opfilt import opfilt_base
from delensalot.core.iterator import bfgs, steps, loggers

alm2rlm = lambda alm : alm # get rid of this
rlm2alm = lambda rlm : rlm


@log_on_start(logging.INFO, " Start of prt_time()")
@log_on_end(logging.INFO, " Finished prt_time()")
def prt_time(dt, label=''):
    dh = np.floor(dt / 3600.)
    dm = np.floor(np.mod(dt, 3600.) / 60.)
    ds = np.floor(np.mod(dt, 60))
    log.info("\r [" + ('%02d:%02d:%02d' % (dh, dm, ds)) + "] " + label)
    return

typs = ['T', 'QU', 'TQU']



class qlm_iterator(object):
    def __init__(self, lib_dir:str, h:str, lm_max_dlm:tuple,
                 dat_maps:list or np.ndarray, plm0:np.ndarray, pp_h0:np.ndarray,
                 cpp_prior:np.ndarray, cls_filt:dict,
                 ninv_filt:opfilt_base.alm_filter_wl,
                 k_geom:utils_geom.Geom,
                 chain_descr, stepper:steps.nrstep,
                 logger=None,
                 NR_method=100, tidy=0, verbose=True, soltn_cond=True, wflm0=None, _usethisE=None, 
                 pp_h0s_matrix = None, inv_signal_matrix = None, 
                 plm0_12 = None, sims_lib = None):
        """Lensing map iterator

            The bfgs hessian updates are called 'hlm's and are either in plm, dlm or klm space

            Args:
                h: 'k', 'd', 'p' if bfgs updates act on klm's, dlm's or plm's respectively
                pp_h0: the starting hessian estimate. (cl array, ~ 1 / N0 of the lensing potential)
                cpp_prior: fiducial lensing potential spectrum used for the prior term
                cls_filt (dict): dictionary containing the filter cmb unlensed spectra (here, only 'ee' is required)
                k_geom: lenspyx geometry for once-per-iterations operations (like checking for invertibility etc, QE evals...)
                stepper: custom calculation of NR-step
                wflm0(optional): callable with Wiener-filtered CMB map search starting point

        """
        assert h in ['k', 'p', 'd']
        lmax_qlm, mmax_qlm = lm_max_dlm
        lmax_filt, mmax_filt = ninv_filt.lmax_sol, ninv_filt.mmax_sol

        if mmax_qlm is None: mmax_qlm = lmax_qlm

        self.h = h

        self.lib_dir = lib_dir
        self.cacher = cachers.cacher_npy(lib_dir)
        self.hess_cacher = cachers.cacher_npy(opj(self.lib_dir, 'hessian'))
        self.wf_cacher = cachers.cacher_npy(opj(self.lib_dir, 'wflms'))
        self.blt_cacher = cachers.cacher_npy(opj(self.lib_dir, 'BLT/'))
        if logger is None:
            logger = logger_norms(opj(lib_dir, 'history_increment.txt'))
        self.logger = logger

        self.chain_descr = chain_descr
        self.opfilt = sys.modules[ninv_filt.__module__] # filter module containing the ch-relevant info
        self.stepper = stepper
        self.soltn_cond = soltn_cond

        self.dat_maps = np.array(dat_maps)

        self.hessian_matrix = pp_h0s_matrix #what is this?? H = (R + C^{-1})^{-1}, note 1/N0 is R, units are of the chh_p signal prior
        self.inv_signal_matrix = inv_signal_matrix

        self.chh = cpp_prior[:lmax_qlm+1] * self._p2h(lmax_qlm) ** 2
        self.hh_h0 = cli(pp_h0[:lmax_qlm + 1] * self._h2p(lmax_qlm) ** 2 + cli(self.chh))  #~ (1/Cpp + 1/N0)^-1
        self.hh_h0 *= (self.chh > 0)
        self.lmax_qlm = lmax_qlm
        self.mmax_qlm = mmax_qlm

        self.NR_method = NR_method
        self.tidy = tidy
        self.verbose = verbose

        self.cls_filt = cls_filt
        self.lmax_filt = lmax_filt
        self.mmax_filt = mmax_filt

        self.filter = ninv_filt
        self.k_geom = k_geom

        self.wflm0 = wflm0
        plm_fname = '%s_%slm_it%03d' % ({'p': 'phi', 'o': 'om'}['p'], self.h, 0)

        if not self.cacher.is_cached(plm_fname):
            print("plm0 not cached, caching", flush = True)
            self.cacher.cache(plm_fname, read_map(plm0)) #almxfl(read_map(plm0), self._p2h(self.lmax_qlm), self.mmax_qlm, False))
        
        self.logger.startup(self)

        self._usethisE = _usethisE
        
        self.sims_lib = sims_lib

    def _p2h(self, lmax):
        if self.h == 'p':
            return np.ones(lmax + 1, dtype=float)
        elif self.h == 'k':
            return 0.5 * np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2, dtype=float)
        elif self.h == 'd':
            return np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2), dtype=float)
        else:
            assert 0, self.h + ' not implemented'

    def _h2p(self, lmax): return cli(self._p2h(lmax))

    def hlm2dlm(self, hlm, inplace):
        if self.h == 'd':
            return hlm if inplace else hlm.copy()
        if self.h == 'p':
            h2d = np.sqrt(np.arange(self.lmax_qlm + 1, dtype=float) * np.arange(1, self.lmax_qlm + 2, dtype=float))
        elif self.h == 'k':
            h2d = cli(0.5 * np.sqrt(np.arange(self.lmax_qlm + 1, dtype=float) * np.arange(1, self.lmax_qlm + 2, dtype=float)))
        else:
            assert 0, self.h + ' not implemented'
        if inplace:
            almxfl(hlm, h2d, self.mmax_qlm, True)
        else:
            return  almxfl(hlm, h2d, self.mmax_qlm, False)


    def _sk2plm(self, itr, shift_1 = 0, shift_2 = 0):
        stringa = self.get_shifted_name(shift_1, shift_2)
        sk_fname = lambda k: 'rlm_sn_%s_%s%s' % (k, 'p', stringa)
        rlm = alm2rlm(self.cacher.load('phi_%slm_it000%s'%(self.h, stringa)))
        for i in range(itr):
            rlm += self.hess_cacher.load(sk_fname(i))
        return rlm2alm(rlm)

    def _yk2grad(self, itr, shift_1 = 0, shift_2 = 0):
        shift_string = self.get_shifted_name(shift_1, shift_2)
        yk_fname = lambda k: 'rlm_yn_%s_%s%s' % (k, 'p', shift_string)
        rlm = alm2rlm(self.load_gradient(0, 'p', shift_1 = shift_1, shift_2 = shift_2))
        for i in range(itr):
            rlm += self.hess_cacher.load(yk_fname(i))
        return rlm2alm(rlm)

    def is_iter_done(self, itr, key):
        """Returns True if the iteration 'itr' has been performed already and False if not

        """
        if itr <= 0:
            return self.cacher.is_cached('%s_%slm_it000' % ({'p': 'phi', 'o': 'om'}[key], self.h))
        sk_fname = lambda k: 'rlm_sn_%s_%s' % (k, 'p')
        return self.hess_cacher.is_cached(sk_fname(itr - 1)) #FIXME

    def _is_qd_grad_done(self, itr, key):
        if itr <= 0:
            return self.cacher.is_cached('%slm_grad%slik_it%03d' % (self.h, key.lower(), 0))
        yk_fname = lambda k: 'rlm_yn_%s_%s' % (k, 'p')
        for i in range(itr):
            if not self.hess_cacher.is_cached(yk_fname(i)):
                return False
        return True


    @log_on_start(logging.INFO, "get_template_blm(it={it}) started")
    @log_on_end(logging.INFO, "get_template_blm(it={it}) finished")
    def get_template_blm(self, it, it_e, lmaxb=1024, lmin_plm=1, elm_wf:None or np.ndarray=None, dlm_mod=None, perturbative=False, k='p_p'):
        """Builds a template B-mode map with the iterated phi and input elm_wf

            Args:
                it: iteration index of lensing tracer
                it_e: iteration index of E-tracer (use it_e = it + 1 for matching lensing and E-templates)
                elm_wf: Wiener-filtered E-mode (healpy alm array), if not an iterated solution (it_e will ignored if set)
                lmin_plm: the lensing tracer is zeroed below lmin_plm
                lmaxb: the B-template is calculated up to lmaxb (defaults to lmax elm_wf)
                perturbative: use pertubative instead of full remapping if set (may be useful for QE)

            Returns:
                blm healpy array

            Note:
                It can be a real lot better to keep the same L range as the iterations

        """
        cache_cond = (lmin_plm >= 1) and (elm_wf is None)

        fn_blt = 'blt_p%03d_e%03d_lmax%s'%(it, it_e, lmaxb)
        fn_blt += '_dlmmod' * dlm_mod.any()
        fn_blt += 'perturbative' * perturbative
        
        if self.blt_cacher.is_cached(fn_blt):
            return self.blt_cacher.load(fn_blt)
        if elm_wf is None:
            if it_e > 0:
                e_fname = 'wflm_%s_it%s' % ('p', it_e - 1)
                assert self.wf_cacher.is_cached(e_fname)
                elm_wf = self.wf_cacher.load(e_fname)
            elif it_e == 0:
                elm_wf = self.wflm0()
            else:
                assert 0,'dont know what to do with it_e = ' + str(it_e)
        if len(elm_wf) == 2 and k == 'p':
            elm_wf = elm_wf[1]
        assert Alm.getlmax(elm_wf.size, self.mmax_filt) == self.lmax_filt, "{}, {}, {}, {}".format(elm_wf.size, self.mmax_filt, Alm.getlmax(elm_wf.size, self.mmax_filt), self.lmax_filt)
        mmaxb = lmaxb
        dlm = self.get_hlm(it, 'p')

        # subtract field from phi
        if dlm_mod is not None:
            dlm = dlm - dlm_mod

        self.hlm2dlm(dlm, inplace=True)
        almxfl(dlm, np.arange(self.lmax_qlm + 1, dtype=int) >= lmin_plm, self.mmax_qlm, True)

        ffi = self.filter.operators.ffi.change_dlm([dlm, None], self.mmax_qlm)
        elm, blm = ffi.lensgclm(np.array([elm_wf, np.zeros_like(elm_wf)]), self.mmax_filt, 2, lmaxb, mmaxb)

        if cache_cond:
            self.blt_cacher.cache(fn_blt, blm)

        return blm

    def get_hlm(self, itr, key, shift_1 = 0, shift_2 = 0):
        """Loads current estimate """
        if itr < 0:
            return np.zeros(Alm.getsize(self.lmax_qlm, self.mmax_qlm), dtype=complex)
        
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.

        stringa = self.get_shifted_name(shift_1, shift_2)

        fn = '%s_%slm_it%03d%s' % ({'p': 'phi', 'o': 'om'}[key.lower()], self.h, itr, stringa)
        if self.cacher.is_cached(fn):
            print("Loading", fn)
            return self.cacher.load(fn)
        return self._sk2plm(itr)


    def load_soltn(self, itr, key):
        """Load starting point for the conjugate gradient inversion.

        """
        assert key.lower() in ['p', 'o']
        for i in np.arange(itr - 1, -1, -1):
            fname = 'wflm_%s_it%s' % (key.lower(), i)
            if self.wf_cacher.is_cached(fname):
                return self.wf_cacher.load(fname), i
        if callable(self.wflm0):
            return self.wflm0(), -1
        # TODO: for MV this need a change
        return np.zeros((1, Alm.getsize(self.lmax_filt, self.mmax_filt)), dtype=complex).squeeze(), -1


    def load_graddet(self, itr, key, shift_1 = 0, shift_2 = 0):
        fn= '%slm_grad%sdet_it%03d' % (self.h, key.lower(), itr)
        return self.cacher.load(fn)

    def load_gradpri(self, itr, key, shift_1 = 0, shift_2 = 0):
        assert key in ['p'], key + ' not implemented'
        assert self.is_iter_done(itr -1 , key)
        ret = self.get_hlm(itr, key, shift_1, shift_2)
        #almxfl(ret, cli(self.chh), self.mmax_qlm, True)
        ret = self.matrix_multiplication(self.inv_signal_matrix, ret)
        return ret

    def load_gradquad(self, k, key, shift_1 = 0, shift_2 = 0):
        stringa = self.get_shifted_name(shift_1, shift_2)
        fn = '%slm_grad%slik_it%03d%s' % (self.h, key.lower(), k, stringa)
        return self.cacher.load(fn)

    def load_gradient(self, itr, key, shift_1 = 0, shift_2 = 0):
        """Loads the total gradient at iteration iter.

                All necessary alm's must have been calculated previously

        """
        if itr == 0:
            g  = self.load_gradpri(0, key, shift_1, shift_2)
            #g += self.load_graddet(0, key, shift_1, shift_2)
            g += self.load_gradquad(0, key, shift_1, shift_2)
            return g
        return self._yk2grad(itr, shift_1, shift_2)

    def calc_norm(self, qlm:np.ndarray):
        return np.sqrt(self.dotop(qlm, qlm))


    def dotop(self, glms1:np.ndarray, glms2:np.ndarray):
        Ncomponents = len(self.filter.operators)
        ret = 0.
        for g1, g2, in zip(np.split(glms1, Ncomponents), np.split(glms2, Ncomponents)):
            cl = alm2cl(g1, g2, None, self.mmax_qlm, None)
            ret += np.sum(cl * (2 * np.arange(len(cl)) + 1))
        return ret

    def matrix_multiplication(self, A, b):

        Ncomponents = len(self.filter.operators)

        ls = np.arange(self.lmax_qlm + 1)
        filter_2 = np.ones_like(ls)
        filter_2[:1] = 0 #some small hack

        ret = np.concatenate([np.sum([almxfl(g, A[:, j, i]*filter_2, self.mmax_qlm, False) for i, g in enumerate(np.split(b, Ncomponents))], axis = 0) for j in range(Ncomponents)])

        return ret

    def apply_H0k(self, grad_lm:np.ndarray, kr):
        """
        Apply (R+C^{-1})^{-1}
        """
        print("Apply custom inverse response for zero step, to normalize the gradient for QE.")

        ret = self.matrix_multiplication(self.hessian_matrix, grad_lm)

        assert ret.shape == grad_lm.shape, (ret.shape, grad_lm.shape)

        #the ret are normalized. Then, I would like to apply a Wiener filter.
        return ret

    @log_on_start(logging.INFO, "get_hessian(k={k}, key={key}) started")
    @log_on_end(logging.INFO, "get_hessian(k={k}, key={key}) finished")
    def get_hessian(self, k, key):
        """Inverse hessian that will produce phi_iter.


        """
        # Zeroth order inverse hessian :
        apply_B0k = lambda rlm, kr: None
        BFGS_H = bfgs.BFGS_Hessian(self.hess_cacher, self.apply_H0k, {}, {}, self.dotop,
                                   L=self.NR_method, verbose=self.verbose, apply_B0k=apply_B0k)
        # Adding the required y and s vectors :
        for k_ in range(np.max([0, k - BFGS_H.L]), k):
            BFGS_H.add_ys('rlm_yn_%s_%s' % (k_, key), 'rlm_sn_%s_%s' % (k_, key), k_)
        return BFGS_H


    @log_on_start(logging.INFO, "build_incr(it={it}, key={key}) started")
    @log_on_end(logging.INFO, "build_incr(it={it}, key={key}) finished")
    def build_incr(self, it, key, gradn, shift_1 = 0, shift_2 = 0):
        """Search direction

           BGFS method with 'self.NR method' BFGS updates to the hessian.
            Initial hessian are built from N0s.

            :param it: current iteration level. Will produce the increment to phi_{k-1}, from gradient est. g_{k-1}
                      phi_{k_1} + output = phi_k
            :param key: 'p' or 'o'
            :param gradn: current estimate of the gradient (alm array)
            :return: increment for next iteration (alm array)

            s_k = x_k+1 - x_k = - H_k g_k
            y_k = g_k+1 - g_k
        """
        assert it > 0, it
        k = it - 2
        stringa = ""
        yk_fname = 'rlm_yn_%s_%s%s' % (k, key, stringa)
        if k >= 0 and not self.hess_cacher.is_cached(yk_fname):  # Caching hessian BFGS yk update :
            yk = alm2rlm(gradn - self.load_gradient(k, key)) #, shift_1 = shift_1, shift_2 = shift_2))
            self.hess_cacher.cache(yk_fname, yk)
        k = it - 1
        BFGS = self.get_hessian(k, key)  # Constructing L-BFGS hessian
        # get descent direction sk = - H_k gk : (rlm array). Will be cached directly
        stringa = self.get_shifted_name(shift_1, shift_2)
        sk_fname = 'rlm_sn_%s_%s%s' % (k, key, stringa)
        if not self.hess_cacher.is_cached(sk_fname):
            log.info("calculating descent direction" )
            t0 = time.time()
            incr = BFGS.get_mHkgk(alm2rlm(gradn), k)
            incr = alm2rlm(self.stepper.build_incr(incr, it, ncomps = len(self.filter.operators)))
            self.hess_cacher.cache(sk_fname, incr)
            prt_time(time.time() - t0, label=' Exec. time for descent direction calculation')
        assert self.hess_cacher.is_cached(sk_fname), sk_fname


    @log_on_start(logging.INFO, "iterate(it={itr}, key={key}) started")
    @log_on_end(logging.INFO, "iterate(it={itr}, key={key}) finished")
    def iterate(self, itr, key):
        """Performs iteration number 'itr'

            This is done by collecting the gradients at level iter, and the lower level potential

        """
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        if not self.is_iter_done(itr, key):
            assert self.is_iter_done(itr - 1, key), 'previous iteration not done'
            self.logger.on_iterstart(itr, key, self)

            glm = self.calc_gradlik(itr, key, shift = 0)
            
            glm += self.calc_graddet(itr, key)
            glm += self.load_gradpri(itr - 1, key)

            glm_list_out = [glm]
            shifts = [0]
            for i, glm in enumerate(glm_list_out):
                glm_incr = np.concatenate([almxfl(g, self.chh > 0, self.mmax_qlm, False) for g in np.split(glm, len(self.filter.operators))])
                self.build_incr(itr, key, glm_incr, shift_1 = shifts[i], shift_2 = shifts[i])

            self.logger.on_iterdone(itr, key, self)
            if self.tidy > 2:  # Erasing deflection databases
                if os.path.exists(opj(self.lib_dir, 'ffi_%s_it%s'%(key, itr))):
                    shutil.rmtree(opj(self.lib_dir, 'ffi_%s_it%s'%(key, itr)))


    @log_on_start(logging.INFO, "calc_gradlik(it={itr}, key={key}) started")
    @log_on_end(logging.INFO, "calc_gradlik(it={itr}, key={key}) finished")
    def calc_gradlik(self, itr, key, iwantit=False, shift = 0):
        """Computes the quadratic part of the gradient for plm iteration 'itr'

        """
        assert self.is_iter_done(itr - 1, key)
        assert itr > 0, itr
        assert key.lower() in ['p', 'o'], key  # potential or curl potential.
        if not self._is_qd_grad_done(itr, key) or iwantit:
            assert key in ['p'], key + '  not implemented'
            #print("Calculating gradlik using pin", flush = True)
            dlm = self.get_hlm(itr - 1, key)

            number_of_operators = len(self.filter.operators)


            #GENERALIZE
            #make the operator set stuff
        
            for opindex, value in enumerate(np.split(dlm, number_of_operators)):
                which = self.filter.operators[opindex].name
                print("Getting", which)
                if which == "p":
                    dlm_temp = value.copy()
                    self.hlm2dlm(dlm_temp, True)

                    dlm_temp_o = None

                    if  "o" in self.filter.operators.names:
                        oindex = self.filter.operators.names.index("o")
                        dlm_temp_o = np.split(dlm, number_of_operators)[oindex]
                        self.hlm2dlm(dlm_temp_o, True) 

                    ffi = self.filter.operators.get("p").field.change_dlm([dlm_temp, dlm_temp_o], self.mmax_qlm, cachers.cacher_mem(safe=False))
                    self.filter.operators.set_field(ffi, which = "p")

                elif (which == "o"):
                    if ("p" not in self.filter.operators.names):
                        dlm_temp = value.copy()
                        self.hlm2dlm(dlm_temp, True)
                        ffi = self.filter.operators.get("o").field.change_dlm([None, dlm_temp], self.mmax_qlm, cachers.cacher_mem(safe=False))
                        self.filter.operators.set_field(ffi, which = "o")
                    else:
                        print("Skipping this.")

                elif which == "a":
                    alpha_map = self.filter.ninv_geom.synthesis(dlm.copy(), spin = 0, lmax = self.lmax_qlm, mmax = self.mmax_qlm, nthreads = 128).squeeze()
                    self.filter.operators.set_field(alpha_map, which = "a")
                elif which == "f":
                    tau_map = self.filter.ninv_geom.synthesis(dlm.copy(), spin = 0, lmax = self.lmax_qlm, mmax = self.mmax_qlm, nthreads = 128).squeeze()
                    self.filter.operators.set_field(tau_map, which = "f")
                else:
                    raise ValueError


            mchain = multigrid.multigrid_chain(self.opfilt, self.chain_descr, self.cls_filt, self.filter)
            if self._usethisE is not None:
                if callable(self._usethisE):
                    log.info("iterator: using custom WF E")
                    soltn = self._usethisE(self.filter, itr)
                else:
                    assert 0, 'dont know what to do this with this E input'
            else:
                soltn, it_soltn = self.load_soltn(itr, key)
                if it_soltn < itr - 1:
                    soltn *= self.soltn_cond
                    mchain.solve(soltn, self.dat_maps, dot_op=self.filter.dot_op())
                    fn_wf = 'wflm_%s_it%s' % (key.lower(), itr - 1)
                    log.info("caching "  + fn_wf)
                    self.wf_cacher.cache(fn_wf, soltn)
                else:
                    log.info("Using cached WF solution at iter %s "%itr)


            t0 = time.time()

            q_geom = pbdGeometry(self.k_geom, pbounds(0., 2 * np.pi))
            
            shift_pairs = [(shift, shift)]
            data_pairs = [(self.dat_maps, soltn)]

            G_total = []

            #TODO, GENERALIZE, NO NEED TO DO THIS
            #taking the gradient should not happen like this

            for o in self.filter.operators:
                which = o.name
                if which == 'p':
                    for index_shift, (data, soltn_) in enumerate(data_pairs):

                        G, C = self.filter.get_qlms(data, soltn_, q_geom, which = which)
                        almxfl(G, self._h2p(self.lmax_qlm), self.mmax_qlm, True)
                        almxfl(C, self._h2p(self.lmax_qlm), self.mmax_qlm, True)
                        G_total.append(G)

                        if "o" in self.filter.operators.names:
                            G_total.append(C)

                elif which == 'o':
                    if "p" not in self.filter.operators.names:
                        G, C = self.filter.get_qlms(self.dat_maps, soltn, q_geom, which = which, shift_1 = shift_1, shift_2 = shift_2)
                        almxfl(C, self._h2p(self.lmax_qlm), self.mmax_qlm, True)
                    else:
                        print("Skip gradient o, already added.")
                else:
                    for index_shift, (shift_1, shift_2) in enumerate(shift_pairs):
                        G = self.filter.get_qlms(self.dat_maps, soltn, q_geom, which = which, shift_1 = shift_1, shift_2 = shift_2)
                        G_total.append(G)


            G = np.concatenate(G_total)
            log.info('get_qlms calculation done; (%.0f secs)'%(time.time() - t0))

            if itr == 1: #We need the gradient at 0 and the yk's to be able to rebuild all gradients
                stringa = self.get_shifted_name(shift, shift)
                fn_lik = '%slm_grad%slik_it%03d%s' % (self.h, key.lower(), 0, stringa)
                self.cacher.cache(fn_lik, -G)

            return -G

        
    def get_shifted_name(self, shift_1, shift_2):
        stringa = ""
        if (shift_1 != 0) or (shift_2 != 0):
            stringa = '_%d_%d' % (shift_1, shift_2)
        return stringa

    @log_on_start(logging.INFO, "calc_graddet(it={itr}, key={key}) started, subclassed")
    @log_on_end(logging.INFO, "calc_graddet(it={itr}, key={key}) finished, subclassed")
    def calc_graddet(self, itr, key):
        assert 0, 'subclass this'

       
class iterator_cstmf(qlm_iterator):
    """Constant mean-field
    """

    def __init__(self, lib_dir:str, h:str, lm_max_dlm:tuple,
                 dat_maps:list or np.ndarray, plm0:np.ndarray, mf0:np.ndarray, pp_h0:np.ndarray,
                 cpp_prior:np.ndarray, cls_filt:dict, ninv_filt:opfilt_base.alm_filter_wl, k_geom:utils_geom.Geom,
                 chain_descr, stepper:steps.nrstep, **kwargs):
        super(iterator_cstmf, self).__init__(lib_dir, h, lm_max_dlm, dat_maps, plm0, pp_h0, cpp_prior, cls_filt,
                                             ninv_filt, k_geom, chain_descr, stepper, **kwargs)
        Ncomponents = len(self.filter.operators)
        mf0_ = np.split(mf0, Ncomponents)[0]
        assert self.lmax_qlm == Alm.getlmax(mf0_.size, self.mmax_qlm), (self.lmax_qlm, Alm.getlmax(mf0_.size, self.lmax_qlm))
        self.cacher.cache('mf', mf0) #almxfl(mf0, self._h2p(self.lmax_qlm), self.mmax_qlm, False))


    @log_on_start(logging.INFO, "load_graddet(it={itr}, key={key}) started")
    @log_on_end(logging.INFO, "load_graddet(it={itr}, key={key}) finished")
    def load_graddet(self, itr, key, shift_1 = 0, shift_2 = 0):
        return self.calc_graddet(itr, key)

    @log_on_start(logging.INFO, "calc_graddet(it={itr}, key={key}) started")
    @log_on_end(logging.INFO, "calc_graddet(it={itr}, key={key}) finished")
    def calc_graddet(self, itr, key):
        #GENERALIZE
        if True:
            from delensalot.biases import grads_mf
            base = 1000
            Nmf = 2
            simidxs = np.arange(base, base+Nmf)
            return grads_mf.load_calculated_mf(self, itr, simidxs, key)
        return 0
    

class logger_norms(loggers.logger_norms):
    def __init__(self, txt_file):
        super().__init__(txt_file)
        self.txt_file = txt_file
        self.ti = None


    def on_iterdone(self, itr:int, key:str, iterator:qlm_iterator):
        incr = iterator.hess_cacher.load('rlm_sn_%s_%s' % (itr-1, key))
        norm_inc = iterator.calc_norm(incr) / iterator.calc_norm(iterator.get_hlm(0, 'p'))
        norms = [iterator.calc_norm(iterator.load_gradient(itr - 1, 'p'))]
        norm_grad_0 = iterator.calc_norm(iterator.load_gradient(0, 'p'))
        for i in [0]: norms[i] = norms[i] / norm_grad_0

        with open(opj(iterator.lib_dir, 'history_increment.txt'), 'a') as file:
            file.write('%03d %.1f %.6f %.6f \n'
                       % (itr, time.time() - self.ti, norm_inc, norms[0]))
            file.close()