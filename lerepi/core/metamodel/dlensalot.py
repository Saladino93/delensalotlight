#!/usr/bin/env python

"""dlensalot.py: Contains the metamodel of the Dlensalot formalism.
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"
# TODO I would like to come up with a better structure for this whole 'DLENSALOT_Model'

import abc
import attr


class DLENSALOT_Concept:
    """An abstract element base type for the Dlensalot formalism."""
    __metaclass__ = abc.ABCMeta


@attr.s
class DLENSALOT_Model(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        data: 
    """
    job = attr.ib(default='')
    data = attr.ib(default='')
    iteration  = attr.ib(default=[])
    geometry = attr.ib(default=[])
    chain_descriptor = attr.ib(default=[])
    stepper = attr.ib(default='')
    map_delensing = attr.ib(default='')
    obd = attr.ib(default='')


@attr.s
class DLENSALOT_Job(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        QE_delensing:
    """
    QE_lensrec = attr.ib(default='')
    MAP_lensrec = attr.ib(default='')
    Btemplate_per_iteration = attr.ib(default='')
    inspect_result = attr.ib(default='')
    map_delensing = attr.ib(default='')
    build_OBD = attr.ib(default='')


@attr.s
class DLENSALOT_Data(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        DATA_LIBDIR: path to the data
    """
    TEMP_suffix = attr.ib(default='')
    fg = attr.ib(default='')
    sims = attr.ib(default='')
    nside = attr.ib(default='')
    BEAM = attr.ib(default='')
    lmax_transf = attr.ib(default='')
    transf = attr.ib(default='')
    zbounds = attr.ib(default='')
    zbounds_len = attr.ib(default='')
    pbounds = attr.ib(default='')
    OBD_type = attr.ib(default='')
    tpl = attr.ib(default='')


@attr.s
class DLENSALOT_Chaindescriptor(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        p0: 
    """
    p0 = attr.ib(default='')
    p1 = attr.ib(default='')
    p2 = attr.ib(default='')
    p3 = attr.ib(default='')
    p4 = attr.ib(default='')
    p5 = attr.ib(default='')
    p6 = attr.ib(default='')
    p7 = attr.ib(default='')


@attr.s
class DLENSALOT_Stepper(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        typ:
    """
    typ = attr.ib(default='')
    lmax_qlm = attr.ib(default='')
    mmax_qlm = attr.ib(default='')
    xa = attr.ib(default='')
    xb = attr.ib(default='')


@attr.s
class DLENSALOT_Iteration(DLENSALOT_Concept):
    """_summary_

    Args:
        DLENSALOT_Concept (_type_): _description_
    """
    K = attr.ib(default='')
    # version, can be 'noMF
    V = attr.ib(default='')
    ITMAX = attr.ib(default='')
    IMIN = attr.ib(default='')
    IMAX = attr.ib(default='')
    # Change the following block only if a full, Planck-like QE lensing power spectrum analysis is desired
    # This uses 'ds' and 'ss' QE's, crossing data with sims and sims with other sims.
    # This remaps idx -> idx + 1 by blocks of 60 up to 300. This is used to remap the sim indices for the 'MCN0' debiasing term in the QE spectrum
    QE_LENSING_CL_ANALYSIS = attr.ib(default='')
    # Change the following block only if exotic transferfunctions are desired
    STANDARD_TRANSFERFUNCTION = attr.ib(default='')
    # Change the following block only if other than cinv_t, cinv_p, ivfs filters are desired
    FILTER = attr.ib(default='')
    # Change the following block only if exotic chain descriptor are desired
    CHAIN_DESCRIPTOR = attr.ib(default='')
    # Change the following block only if other than sepTP for QE is desired
    FILTER_QE = attr.ib(default='')
    # Choose your iterator. Either pertmf or const_mf
    ITERATOR = attr.ib(default='')
    # The following block defines various multipole limits. Change as desired
    lmax_filt = attr.ib(default='') # unlensed CMB iteration lmax
    lmin_tlm = attr.ib(default='')
    lmin_elm = attr.ib(default='')
    lmin_blm = attr.ib(default='')
    lmax_qlm = attr.ib(default='')
    mmax_qlm = attr.ib(default='')
    lmax_unl = attr.ib(default='')
    mmax_unl = attr.ib(default='')
    lmax_ivf = attr.ib(default='')
    mmax_ivf = attr.ib(default='')
    lmin_ivf = attr.ib(default='')
    mmin_ivf = attr.ib(default='')
    LENSRES = attr.ib(default='') # Deflection operations will be performed at this resolution
    Lmin = attr.ib(default='') # The reconstruction of all lensing multipoles below that will not be attempted
    # Meanfield, OBD, and tol settings
    CG_TOL = attr.ib(default='')
    TOL = attr.ib(default='')
    soltn_cond = attr.ib(default='')
    nsims_mf = attr.ib(default='')
    OMP_NUM_THREADS = attr.ib(default='')


@attr.s
class DLENSALOT_Geometry(DLENSALOT_Concept):
    """_summary_

    Args:
        DLENSALOT_Concept (_type_): _description_
    """
    lmax_unl = attr.ib(default='')
    zbounds = attr.ib(default='')
    zbounds_len = attr.ib(default='')
    pbounds = attr.ib(default='')
    nside = attr.ib(default='')
    lenjob_geometry = attr.ib(default='')
    lenjob_pbgeometry = attr.ib(default='')
    ninvjob_geometry = attr.ib(default='')
    ninvjob_qe_geometry = attr.ib(default='')


@attr.s
class DLENSALOT_Mapdelensing(DLENSALOT_Concept):
    """_summary_

    Args:
        DLENSALOT_Concept (_type_): _description_
    """
    edges = attr.ib(default='')
    IMIN = attr.ib(default='')
    IMAX = attr.ib(default='')
    ITMAX = attr.ib(default='')
    fg = attr.ib(default='')
    base_mask = attr.ib(default='')
    nlevels = attr.ib(default='')
    nside = attr.ib(default='')
    lmax_cl = attr.ib(default='')
    beam = attr.ib(default='')
    lmax_transf = attr.ib(default='')
    transf = attr.ib(default='')
    Cl_fid = attr.ib(default='')


@attr.s
class DLENSALOT_OBD(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        typ:
    """
    nlev_dep = attr.ib(default='')
    inf = attr.ib(default='')
    ratio = attr.ib(default='')
    BMARG_LIBDIR = attr.ib(default='')
    BMARG_LIBDIR_buildpath = attr.ib(default='')
    BMARG_LCUT = attr.ib(default='')
    BMARG_RESCALE = attr.ib(default='')
    CENTRALNLEV_UKAMIN = 0.42,
    nlev_t = 0.42/np.sqrt(2),
    nlev_p = 0.42
    nlev_dep = 10000.,
    inf = 1e4,
    ratio = np.inf,
    mask = ('nlev', 100),
    noisemodel_rhits = '/global/project/projectdirs/cmbs4/awg/lowellbb/reanalysis/mapphi_intermediate/s08b/masks/08b_rhits_positive_nonan.fits', #If OBD used, this must be the exact same map with which tniti was build
    noisemodel_norm = 1.0, #divide noisemodel by this value # TODO not sure if needed