import numpy as np

from delensalot.utility.utils_hp import almxfl
from delensalot.utils import cli
from delensalot.utils import read_map

class nrstep(object):
    def __init__(self, lmax_qlm:int, mmax_qlm:int, val=1., vals = None):
        self.lmax_qlm = lmax_qlm
        self.mmax_qlm = mmax_qlm
        self.val = val
        self.ones = np.ones(self.lmax_qlm + 1, dtype=float)
        self.filt = lambda x: self.ones*x
        self.vals = vals

    def steplen(self, itr, incrnorm):
        return self.val

    #def build_incr(self, incrlm, itr):
    #    print('incr step val %.5f'%self.val)
    #    return incrlm * self.val
    def build_incr(self, incrlm, itr, ncomps = 1):
        fl = self.steplen(itr, incrlm)
        if ncomps == 1:
            incrlm = almxfl(incrlm, fl * self.filt(self.val), self.mmax_qlm, False)
        else:
            incrlm = np.concatenate([almxfl(incr, fl * self.filt(self.vals[idx]), self.mmax_qlm, False) for idx, incr in enumerate(np.split(incrlm, ncomps))])
        return incrlm

class harmonicbump(nrstep):
    def __init__(self, lmax_qlm, mmax_qlm, xa=400, xb=1500, a=0.5, b=0.1, scale=50, flt=None):
        """Harmonic bumpy step that were useful for s06b and s08b

        """
        super().__init__(lmax_qlm, mmax_qlm)
        filt = np.ones(self.lmax_qlm + 1, dtype=float)
        if flt is not None:
            filt[:min(len(flt), lmax_qlm+1)] = flt[:min(len(flt), lmax_qlm+1)]
        self.scale = scale
        self.bump_params = (xa, xb, a, b)
        self.filt = filt

    def steplen(self, itr, incrnorm):
        xa, xb, a, b = self.bump_params
        return self.bp(np.arange(self.lmax_qlm + 1),xa, a, xb, b, scale=self.scale)


    def build_incr(self, incrlm, itr, ncomps = 1):
        fl = self.steplen(itr, incrlm)
        if ncomps == 1:
            almxfl(incrlm, fl * self.filt, self.mmax_qlm, True)
        else:
            incrlm = np.concatenate([almxfl(incr, fl * self.filt, self.mmax_qlm, False) for incr in np.split(incrlm, ncomps)])
        return incrlm

    @staticmethod
    def bp(x, xa, a, xb, b, scale=50):
            """Bump function with f(xa) = a and f(xb) =  b with transition at midpoint over scale scale

            """
            x0 = (xa + xb) * 0.5
            r = lambda x_: np.arctan(np.sign(b - a) * (x_ - x0) / scale) + np.sign(b - a) * np.pi * 0.5
            return a + r(x) * (b - a) / r(xb)

# class hmapprescal(nrstep):
#     import scarf
#     from delensalot.core.helper import utils_scarf
#     def __init__(self, lmax_qlm:int, mmax_qlm:int, incr2klm:np.ndarray, hmmap, valrange:tuple, geom:utils_scarf.Geometry, sht_threads:int):
#         """The increment follows here some 'hitmap'. Points close to the mask are forced to move slowly

#             Args:
#                 incr2klm: harmonic filter to apply to the increment, before real-space rescaling
#                 hmmap: the map (or its path) used to rescale the increment (must be compatible with input geom)
#                 valrange (two floats): the applied rescaling is ( vmin + (vmax - vmin) * (hmap / np.max(hmap)) )
#                 geom: scarf geometry for the sht transforms

#         """
#         super().__init__(lmax_qlm, mmax_qlm)
#         assert incr2klm.size > self.lmax_qlm
#         self.hmap = hmmap
#         self.valrange = valrange
#         self.incr2klm = incr2klm

#         sc_job = utils_scarf.scarfjob()
#         sc_job.set_geometry(geom)
#         sc_job.set_triangular_alm_info(self.lmax_qlm, self.mmax_qlm)
#         sc_job.set_nthreads(sht_threads)
#         self.sc_job = sc_job

#     def build_incr(self, incrlm, itr):
#         hmap = read_map(self.hmap)
#         increal = self.sc_job.alm2map(almxfl(incrlm, self.incr2klm, self.mmax_qlm, False))
#         vmin, vmax = self.valrange
#         increal *=  ( vmin + (vmax - vmin) * (hmap / np.max(hmap)) )
#         return almxfl(self.sc_job.map2alm(increal), cli(self.incr2klm), self.lmax_qlm, False)

#     def steplen(self, itr, incrnorm):
#         return self.valrange[1]