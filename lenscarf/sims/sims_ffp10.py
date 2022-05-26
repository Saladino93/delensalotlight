import os
import numpy as np
from lenscarf import cachers
from lenscarf import utils_scarf, utils_hp
from plancklens.sims.planck2018_sims import cmb_unl_ffp10
from plancklens import utils
from lenscarf.remapping import deflection

aberration_lbv_ffp10 = (264. * (np.pi / 180), 48.26 * (np.pi / 180), 0.001234)

class cmb_len_ffp10:
    def __init__(self, aberration:tuple[float, float, float]=aberration_lbv_ffp10,cacher: cachers.cacher or None=None,
                       lmax_thingauss=5120, nbands=1, verbose=False):
        """FFP10 lensed cmbs, lensed with independent lenscarf code on thingauss geometry


            Args:
                aberration: aberration parameters (gal. longitude (rad), latitude (rad) and v/c) Defaults to FFP10 values
                cacher: set this to one of lenscarf.cachers in order save maps (nothing saved by default)
                nbands: if set splits the sky into bands to perform the operations (saves some memory but probably a bit slower)

            Note: (FIXME)
                 The calculation of the FFT plans by FFTW can totally dominate if doing only very few remappings in the same session

        """
        nbands = int(nbands + (1 - int(nbands)%2))  # want an odd number to avoid a split just on the equator
        assert nbands <= 10, 'did not check'
        if cacher is None:
            cacher = cachers.cacher_none() # This cacher saves nothing
        self.cacher = cacher

        self.fft_tr = int(os.environ.get('OMP_NUM_THREADS', 1))
        self.sht_tr = int(os.environ.get('OMP_NUM_THREADS', 1))

        self.lmax_len = 4096 # FFP10 lensed CMBs were designed for this lmax
        self.mmax_len = 4096
        self.lmax_thingauss = lmax_thingauss

        self.targetres = 0.75  # Main accuracy parameter. I belive this crudely matches the FFP10 pipeline's

        zls, zus = self._mkbands(nbands)
        # By construction the central one covers the equator
        len_geoms = [utils_scarf.Geom.get_thingauss_geometry(lmax_thingauss, 2, zbounds=(zls[nbands//2], zus[nbands//2]))]
        for ib in range(nbands//2):
            # and the other ones are symmetric w.r.t. the equator
            geom_north = utils_scarf.Geom.get_thingauss_geometry(lmax_thingauss, 2, zbounds=(zls[ib], zus[ib]))
            geom_south = utils_scarf.Geom.get_thingauss_geometry(lmax_thingauss, 2, zbounds=(zls[nbands-ib], zus[nbands-ib]))
            len_geoms.append(utils_scarf.Geom.merge([geom_north, geom_south]))
        pbdGeoms = [utils_scarf.pbdGeometry(len_geom, utils_scarf.pbounds(np.pi, 2 * np.pi)) for len_geom in len_geoms]

        # Sanity check, we cant have rings overlap
        ref_geom = utils_scarf.Geom.get_thingauss_geometry(self.lmax_thingauss, 2)
        tht_all = np.concatenate([pbgeo.geom.theta for pbgeo in pbdGeoms])
        assert np.all(np.sort(ref_geom.theta) == np.sort(tht_all))

        self.pbdGeoms = pbdGeoms
        self.nbands = len(pbdGeoms)

        l, b, v = aberration
        # \phi_{10} = - \sqrt{4\pi/3} n_z
        # \phi_{11} = + \sqrt{4\pi / 3} \frac{(n_x - i n_y)}{\sqrt{2}}
        vlm = np.array([0., np.cos(b), - np.exp(-1j * l) * np.sin(b) / np.sqrt(2.)])  # LM = 00, 10 and 11
        self.vlm = vlm * (-v * np.sqrt(4 * np.pi / 3))

        self.verbose = verbose

    @staticmethod
    def _mkbands(nbands: int):
        """Splits the sky in nbands regions with equal numbers of latitude points """
        thts, dt = np.linspace(0, np.pi, nbands + 1), 0. / 180 / 60 * np.pi  # no buffer here
        th_l = thts[:-1] - dt
        th_u = thts[1:] + dt
        th_l[0] = 0.
        th_u[-1] = np.pi
        zu = np.cos(th_l[::-1])
        zl = np.cos(th_u[::-1])
        zl[0] = -1.
        zu[-1] = 1.
        return zl, zu

    def _get_dlm(self, idx):
        dlm = cmb_unl_ffp10.get_sim_plm(idx)
        dlm[:len(self.vlm)] += self.vlm # aberration
        lmax_dlm = utils_hp.Alm.getlmax(dlm.size, -1)
        mmax_dlm = lmax_dlm
        p2d = np.sqrt(np.arange(lmax_dlm + 1) * np.arange(1, lmax_dlm + 2))
        utils_hp.almxfl(dlm, p2d, mmax_dlm, inplace=True)
        return dlm, lmax_dlm, mmax_dlm

    def _build_eb(self, idx):
        dlm, lmax_dlm, mmax_dlm = self._get_dlm(idx)
        len_eblm = np.zeros((2, utils_hp.Alm.getsize(self.lmax_len, self.mmax_len)), dtype=complex)
        unl_elm = cmb_unl_ffp10.get_sim_elm(idx)
        unl_blm = cmb_unl_ffp10.get_sim_blm(idx)
        lmax_elm = utils_hp.Alm.getlmax(unl_elm.size, -1)
        mmax_elm = lmax_elm
        assert lmax_elm == utils_hp.Alm.getlmax(unl_blm.size, -1)
        for i, pbdGeom in utils.enumerate_progress(self.pbdGeoms, 'collecting bands'):
            ffi = deflection(pbdGeom, self.targetres, dlm, mmax_dlm, self.fft_tr, self.sht_tr, verbose=self.verbose)
            len_eblm += ffi.lensgclm([unl_elm, unl_blm], mmax_elm, 2, self.lmax_len, self.mmax_len)
        return len_eblm

    def get_sim_tlm(self, idx):
        fn = 'tlm_%04d' % idx
        if not self.cacher.is_cached(fn):
            dlm, lmax_dlm, mmax_dlm = self._get_dlm(idx)
            unl_tlm = cmb_unl_ffp10.get_sim_tlm(idx)
            len_tlm = np.zeros(utils_hp.Alm.getsize(self.lmax_len, self.mmax_len), dtype=complex)
            lmax_tlm = utils_hp.Alm.getlmax(unl_tlm.size, -1)
            mmax_tlm = lmax_tlm
            for i, pbdGeom in utils.enumerate_progress(self.pbdGeoms, 'collecting bands'):
                ffi = deflection(pbdGeom, self.targetres, dlm, mmax_dlm, self.fft_tr, self.sht_tr, verbose=self.verbose)
                len_tlm += ffi.lensgclm(unl_tlm, mmax_tlm, 0, self.lmax_len, self.mmax_len)
            self.cacher.cache(fn, len_tlm)
            return len_tlm
        return self.cacher.load(fn)

    def get_sim_eblm(self, idx):
        fn_e = 'elm_%04d' % idx
        fn_b = 'blm_%04d' % idx
        if not self.cacher.is_cached(fn_e) or not self.cacher.is_cached(fn_b):
            len_elm, len_blm = self._build_eb(idx)
            self.cacher.cache(fn_b, len_blm)
            self.cacher.cache(fn_e, len_elm)
            return len_elm, len_blm
        return self.cacher.load(fn_e), self.cacher.load(fn_b)

    def get_sim_elm(self, idx):
        fn_e = 'elm_%04d' % idx
        fn_b = 'blm_%04d' % idx
        if not self.cacher.is_cached(fn_e):
            len_elm, len_blm = self._build_eb(idx)
            self.cacher.cache(fn_b, len_blm)
            self.cacher.cache(fn_e, len_elm)
            return len_elm
        return self.cacher.load(fn_e)

    def get_sim_blm(self, idx):
        fn_e = 'elm_%04d' % idx
        fn_b = 'blm_%04d' % idx
        if not self.cacher.is_cached(fn_b):
            len_elm, len_blm = self._build_eb(idx)
            self.cacher.cache(fn_e, len_elm)
            self.cacher.cache(fn_b, len_blm)
            return len_blm
        return self.cacher.load(fn_b)