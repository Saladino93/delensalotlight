"""Module collecting frequent or useful configurations


"""
import numpy as np
from lenscarf import utils_scarf as us

def cmbs4_06b():
    zbounds_len = [-0.9736165659024625, -0.4721687661208586]
    pbounds_exl = np.array((113.20399439681668, 326.79600560318335)) #These were the pbounds as defined with the old itercurv conv.
    pb_ctr = np.mean([-(360. - pbounds_exl[1]), pbounds_exl[0]])
    pb_extent = pbounds_exl[0] + (360. - pbounds_exl[1])
    scarf_job = us.scarfjob()
    scarf_job.set_healpix_geometry(2048, zbounds=zbounds_len)
    return scarf_job, [pb_ctr / 180 * np.pi, pb_extent/ 180 * np.pi], zbounds_len

def cmbs4_08b_healpix():
    zbounds_len = [-0.9736165659024625, -0.4721687661208586]
    pbounds_exl = np.array((113.20399439681668, 326.79600560318335))
    pb_ctr = np.mean([-(360. - pbounds_exl[1]), pbounds_exl[0]])
    pb_extent = pbounds_exl[0] + (360. - pbounds_exl[1])
    scarf_job = us.scarfjob()
    scarf_job.set_healpix_geometry(2048, zbounds=zbounds_len)
    return scarf_job, [pb_ctr/ 180 * np.pi, pb_extent/ 180 * np.pi], zbounds_len