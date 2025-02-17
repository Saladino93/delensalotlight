"""
Scripts to compute the gradient MF from the MAP lensing field. 

Using or not the trick from Carron and Lewis 2017.

"""
import numpy as np
from delensalot.utility import utils_hp as uhp
from delensalot.utils import clhash, cli
from delensalot.core import cachers
from plancklens.qcinv import multigrid
from plancklens.sims import phas
from delensalot.core.iterator import cs_iterator, statics
from lenspyx.remapping.utils_geom import pbdGeometry, pbounds

import healpy as hp
from os.path import join as opj
import os
import time

from delensalot.utility.utils_hp import almxfl as hp_almxfl, Alm

from lenspyx.utils_hp import alm_copy
from plancklens.utils import alm_copy as alm_copypl

import copy



def fg_phases(mappa: np.ndarray, seed: int = 0):
    np.random.seed(seed)
    f = lambda z: np.exp(1j*np.random.uniform(0., 2.*np.pi, size = z.shape))
    return f(mappa)

def randomize(x, idx = 0, shift = 100):
    return x*fg_phases(x, idx+shift)



def set_operator(itlib, dlm):

    number_of_operators = len(itlib.filter.operators)

    for opindex, value in enumerate(np.split(dlm, number_of_operators)):
        which = itlib.filter.operators[opindex].name
        print("Getting", which)
        if which == "p":
            dlm_temp = value.copy()
            itlib.hlm2dlm(dlm_temp, True)
            dlm_temp_o = None

            if  "o" in itlib.filter.operators.names:
                oindex = itlib.filter.operators.names.index("o")
                dlm_temp_o = np.split(dlm, number_of_operators)[oindex]
                itlib.hlm2dlm(dlm_temp_o, True) 

            ffi = itlib.filter.operators.get("p").field.change_dlm([dlm_temp, dlm_temp_o], itlib.mmax_qlm, cachers.cacher_mem(safe=False))
            itlib.filter.operators.set_field(ffi, which = "p")

        elif (which == "o"):
            if ("p" not in itlib.filter.operators.names):
                dlm_temp = value.copy()
                itlib.hlm2dlm(dlm_temp, True)
                ffi = itlib.filter.operators.get("o").field.change_dlm([None, dlm_temp], itlib.mmax_qlm, cachers.cacher_mem(safe=False))
                itlib.filter.operators.set_field(ffi, which = "o")
            else:
                print("Skipping this.")

        elif which == "a":
            alpha_map = itlib.filter.ninv_geom.synthesis(dlm.copy(), spin = 0, lmax = itlib.lmax_qlm, mmax = itlib.mmax_qlm, nthreads = 24).squeeze()
            itlib.filter.operators.set_field(alpha_map, which = "a")
        elif which == "f":
            tau_map = itlib.filter.ninv_geom.synthesis(dlm.copy(), spin = 0, lmax = itlib.lmax_qlm, mmax = itlib.mmax_qlm, nthreads = 24).squeeze()
            itlib.filter.operators.set_field(tau_map, which = "f")
        else:
            raise ValueError

def get_solution_hlm(index, itr, itlib, recs_folder, apply_zero = False):
    rec = statics.rec()
    lib_dir = recs_folder
    plm = rec.load_plms(lib_dir=lib_dir, itrs = [itr])[0]
    plm *= (1-apply_zero)
    #alm, plm, olm = np.split(plm, 3)

    #dlmsolution = itlib.get_hlm(itr - 1, "p")
    #alm_fixed, plm_fixed, olm_fixed = np.split(dlmsolution, 3)

    alm_fixed = plm #np.concatenate([alm_fixed, plm, olm])
    #plm_fixed = np.concatenate([alm, plm_fixed, olm_fixed])

    set_operator(itlib, alm_fixed)
    filter_alm_fixed = copy.deepcopy(itlib.filter) #COPY itlib X
    
    #set_operator(itlib, plm_fixed)
    #filter_plm_fixed = copy.deepcopy(itlib.filter) #COPY itlib X

    return filter_alm_fixed, 0#filter_plm_fixed
    

def calc_sims_v3(itlib, simidxs, itr, cls_unl, lmax_qlm, mmax_qlm, key:str = 'p', originalidx = 0, recsfolder = None, which = "a"):

    str = "" if which == "a" else f"_{which}"

    print("Original index,", originalidx, "sim indexes", simidxs)

    fn_lik = lambda this_idx : f'{itlib.h}lm_N0{key.lower()}{itr:03d}_v4_vrandnew2_sims_v3{str}_sim{this_idx:04d}'
    
    print("Starting...")

    cacher = itlib.cacher
    q_geom = pbdGeometry(itlib.k_geom, pbounds(0., 2 * np.pi))
    wf_cacher = itlib.wf_cacher
    e_fname = 'wflm_%s_it%s' % ('p', itr - 1)
    wf_data = wf_cacher.load(e_fname)
    data = itlib.dat_maps

    filter_alm_fixed, filter_plm_fixed = get_solution_hlm(originalidx, itr, itlib, recsfolder)
    del filter_plm_fixed

    ###########################################
    d, x, y = "d", "i", "j"

    number_of_operators = len(itlib.filter.operators)

    print("Getting gradients")

    s = time.time()

    for simidx in simidxs:

        if cacher.is_cached(fn_lik(simidx)):
            continue

        print("File Name", fn_lik(simidx))

        shift1 = 1000*(originalidx+1)+simidx
        shift2 = 2000*(originalidx+1)+simidx

        plm0_i_j_A_new = filter_alm_fixed.get_qlms(data, wf_data, q_geom, wf_data, which, filter_leg2 = filter_alm_fixed, shift_1 = shift1, shift_2 = shift2, cache = True)
        if which == "p":
            plm0_i_j_A_new = plm0_i_j_A_new[0]
        plm0_d_j_A_new = filter_alm_fixed.get_qlms(data, wf_data, q_geom, wf_data, which, shift_1 = 0, shift_2 = shift2, cache = True)
        if which == "p":
            plm0_d_j_A_new = plm0_d_j_A_new[0]
        plm0_j_d_A_new = filter_alm_fixed.get_qlms(data, wf_data, q_geom, wf_data, which, shift_1 = shift1, shift_2 = 0, cache = True)
        if which == "p":
            plm0_j_d_A_new = plm0_j_d_A_new[0]
        plm0_j_i_A_new = filter_alm_fixed.get_qlms(data, wf_data, q_geom, wf_data, which, filter_leg2 = filter_alm_fixed, shift_1 = shift2, shift_2 = shift1, cache = True)
        if which == "p":
            plm0_j_i_A_new = plm0_j_i_A_new[0]
        plm0_d_d_A_new = filter_alm_fixed.get_qlms(data, wf_data, q_geom, wf_data, which, shift_1 = 0, shift_2 = 0)

        coadd = (plm0_j_d_A_new+plm0_d_j_A_new)*0.5
        coadd2 = (plm0_j_i_A_new+plm0_i_j_A_new)*0.5

        cls = (4*hp.alm2cl(coadd)-2*hp.alm2cl(coadd2))
        clsymm = hp.alm2cl(plm0_i_j_A_new+plm0_j_i_A_new) 
        cld = hp.alm2cl(plm0_d_d_A_new)
        cacher.cache(fn_lik(simidx), np.c_[cls, cld, clsymm])

    print("Time (s) to get gradients", time.time() - s)
    print("Finished!!!")

    return 0



def calc_sims_v2(itlib, simidx, itr, cls_unl, lmax_qlm, mmax_qlm, key:str = 'p', originalidx = 0, recsfolder = None, which = "a"):

    str = "" if which == "a" else f"_{which}"

    print("Original index", originalidx, "sim index", simidx)

    fn_lik = lambda this_idx : f'{itlib.h}lm_N0{key.lower()}{itr:03d}_v4_vrandnew2_sims_v2{str}_sim{this_idx:04d}'
    
    q_geom = pbdGeometry(itlib.k_geom, pbounds(0., 2 * np.pi))
    cacher = itlib.cacher
    wf_cacher = itlib.wf_cacher
    e_fname = 'wflm_%s_it%s' % ('p', itr - 1)
    wf_data = wf_cacher.load(e_fname)
    data = itlib.dat_maps
    #first, get the current estimate of the MAP
    #dlmsolution = itlib.get_hlm(itr - 1, key)
    #set_operator(itlib, dlmsolution)
    #itlib_filter_sol_d = itlib.filter #COPY itlib X

    filter_alm_fixed, filter_plm_fixed = get_solution_hlm(originalidx, itr, itlib, recsfolder)
    del filter_plm_fixed

    operate = False

    if operate:
        filter_alm_fixed_2, filter_plm_fixed = get_solution_hlm(2*originalidx+1, itr, itlib)
        del filter_plm_fixed
    else:
        filter_alm_fixed_2 = filter_alm_fixed

    ###########################################
    d, x, y = "d", "i", "j"

    print("File Name", fn_lik(simidx))

    if cacher.is_cached(fn_lik(simidx)):
        return cacher.load(fn_lik(simidx))

    print("Starting...")

    number_of_operators = len(itlib.filter.operators)

    #get phases X
    phases_x = 1
    #dlm = np.concatenate([x*phases_x for x in np.split(dlmsolution, number_of_operators)], axis = 0) #original fields with phases x
    #set_operator(itlib, dlm)
    eblm_dat_i_alm_fixed = filter_alm_fixed.synalm(cls_unl, seed = simidx+1)
    #dlm = np.concatenate([x*phases_x for x in np.split(dlmsolution, number_of_operators)], axis = 0) #original solution with phases x
    #set_operator(itlib, dlm)
    #itlib_filter_sol_x = itlib.filter 
    mchain = multigrid.multigrid_chain(itlib.opfilt, itlib.chain_descr, itlib.cls_filt, filter_alm_fixed)
    elm_wf_i = np.zeros(Alm.getsize(itlib.filter.lmax_sol, itlib.filter.mmax_sol), dtype=complex)
    mchain.solve(elm_wf_i, eblm_dat_i_alm_fixed, dot_op=itlib.filter.dot_op())

    if True:
        #get phases Y
        phases_y = 1
        #dlm = np.concatenate([x*phases_y for x in np.split(dlmsolution, number_of_operators)], axis = 0) #original fields with phases y
        #set_operator(itlib, dlm)    
        #eblm_dat_j = itlib.filter.synalm(cls_unl, seed = simidx+5)
        eblm_dat_j_alm_fixed = filter_alm_fixed_2.synalm(cls_unl, seed = 2*simidx+5)
        #dlm = np.concatenate([x*phases_y for x in np.split(dlmsolution, number_of_operators)], axis = 0) #original solution with phases y
        #set_operator(itlib, dlm)
        #itlib_filter_sol_y = itlib.filter 
        mchain = multigrid.multigrid_chain(itlib.opfilt, itlib.chain_descr, itlib.cls_filt, filter_alm_fixed_2)
        elm_wf_j = np.zeros(Alm.getsize(itlib.filter.lmax_sol, itlib.filter.mmax_sol), dtype=complex)
        mchain.solve(elm_wf_j, eblm_dat_j_alm_fixed, dot_op=itlib.filter.dot_op())
    else:
        elm_wf_j = elm_wf_i
        eblm_dat_j_alm_fixed = eblm_dat_i_alm_fixed


    if operate:
        ## let's create two other legs
        filter_alm_fixed_A, filter_plm_fixed = get_solution_hlm(originalidx+1, itr, itlib)
        del filter_plm_fixed
        filter_alm_fixed_2_A, filter_plm_fixed = get_solution_hlm(2*originalidx+2, itr, itlib)
        del filter_plm_fixed

        eblm_dat_i_alm_fixed_A = filter_alm_fixed_A.synalm(cls_unl, seed = simidx)
        mchain = multigrid.multigrid_chain(itlib.opfilt, itlib.chain_descr, itlib.cls_filt, filter_alm_fixed_A)
        elm_wf_i_A = np.zeros(Alm.getsize(itlib.filter.lmax_sol, itlib.filter.mmax_sol), dtype=complex)
        mchain.solve(elm_wf_i_A, eblm_dat_i_alm_fixed_A, dot_op=itlib.filter.dot_op())

        eblm_dat_j_alm_fixed_A = filter_alm_fixed_2_A.synalm(cls_unl, seed = 2*simidx+5)
        mchain = multigrid.multigrid_chain(itlib.opfilt, itlib.chain_descr, itlib.cls_filt, filter_alm_fixed_2_A)
        elm_wf_j_A = np.zeros(Alm.getsize(itlib.filter.lmax_sol, itlib.filter.mmax_sol), dtype=complex)
        mchain.solve(elm_wf_j_A, eblm_dat_j_alm_fixed_A, dot_op=itlib.filter.dot_op())
    else:
        elm_wf_i_A = elm_wf_i
        elm_wf_j_A = elm_wf_j
        eblm_dat_i_alm_fixed_A = eblm_dat_i_alm_fixed
        filter_alm_fixed_2_A = filter_alm_fixed
        filter_alm_fixed_A = filter_alm_fixed

    print("Getting gradients")
    plm0_i_j_A_new = filter_alm_fixed.get_qlms(eblm_dat_i_alm_fixed, elm_wf_i, q_geom, elm_wf_j, which, filter_leg2 = filter_alm_fixed)
    plm0_d_j_A_new = filter_alm_fixed.get_qlms(data, wf_data, q_geom, elm_wf_j, which)
    plm0_d_d_A_new = filter_alm_fixed.get_qlms(data, wf_data, q_geom, wf_data, which)
    #cacher.cache(fn_lik(simidx, d, y), np.array(G))
    plm0_j_d_A_new = filter_alm_fixed.get_qlms(eblm_dat_j_alm_fixed, elm_wf_j, q_geom, wf_data, which)
    plm0_j_i_A_new = filter_alm_fixed_2.get_qlms(eblm_dat_j_alm_fixed, elm_wf_j, q_geom, elm_wf_i, which, filter_leg2 = filter_alm_fixed)

    if which == "p":
        plm0_i_j_A_new = plm0_i_j_A_new[0]
        plm0_d_j_A_new = plm0_d_j_A_new[0]
        plm0_j_d_A_new = plm0_j_d_A_new[0]
        plm0_j_i_A_new = plm0_j_i_A_new[0]

    coadd = (plm0_j_d_A_new+plm0_d_j_A_new)*0.5
    coadd2 = (plm0_j_i_A_new+plm0_i_j_A_new)*0.5

    cls = (4*hp.alm2cl(coadd)-2*hp.alm2cl(coadd2))
    clsymm = hp.alm2cl(plm0_i_j_A_new+plm0_j_i_A_new) 
    cld = hp.alm2cl(plm0_d_d_A_new)
    cacher.cache(fn_lik(simidx), np.c_[cls, cld, clsymm])

    print("Finished!!!")

    return 0


def calc_sims(itlib, simidx, itr, cls_unl, lmax_qlm, mmax_qlm, key:str = 'p', originalidx = 0):

    fn_lik = lambda this_idx, x, y : f'{itlib.h}lm_N0{key.lower()}{itr:03d}_{x}_{y}_sim{this_idx:04d}'
    q_geom = pbdGeometry(itlib.k_geom, pbounds(0., 2 * np.pi))
    cacher = itlib.cacher
    wf_cacher = itlib.wf_cacher
    e_fname = 'wflm_%s_it%s' % ('p', itr - 1)
    wf_data = wf_cacher.load(e_fname)
    data = itlib.dat_maps
    #first, get the current estimate of the MAP
    dlmsolution = itlib.get_hlm(itr - 1, key)
    set_operator(itlib, dlmsolution)
    itlib_filter_sol_d = copy.deepcopy(itlib.filter) #COPY itlib X
    G_total = []
    for o in itlib.filter.operators:
        which = o.name
        if which == 'p':

            G, C = itlib.filter.get_qlms(data, wf_data, q_geom, which = which) #one leg and another leg
            hp_almxfl(G, itlib.filter._h2p(which, lmax_qlm), mmax_qlm, True)
            hp_almxfl(C, itlib.filter._h2p(which, lmax_qlm), mmax_qlm, True)
            G_total.append(G)

            if "o" in itlib.filter.operators.names:
                G_total.append(C)
        elif which == 'o':
            if "p" not in itlib.filter.operators.names:
                G, C = itlib.filter.get_qlms(data, wf_data, q_geom, which = which)
                hp_almxfl(C, itlib.filter._h2p(which, lmax_qlm), mmax_qlm, True)
            else:
                print("Skip gradient o, already added.")
        else:
            G = itlib.filter.get_qlms(data, wf_data, q_geom, which = which)
            G_total.append(G)

    G_total_data = np.concatenate(G_total)
    cacher.cache(fn_lik(simidx, "d", "d"), np.array(G_total_data))
    del G_total_data

    ###########################################
    x, y = "i", "j"
    if cacher.is_cached(fn_lik(simidx, x, y)) and cacher.is_cached(fn_lik(simidx, y, x)):
        return cacher.load(fn_lik(simidx, x, y)), cacher.load(fn_lik(simidx, y, x))


    number_of_operators = len(itlib.filter.operators)


    #get also some inputs
    dlm_temp = alm_copypl(hp.read_alm(f"/users/odarwish/scratch/JOINTRECONSTRUCTION/apo_new/simswalpha/sim_{originalidx:04d}_plm.fits"), lmax = lmax_qlm)#NOTE
    dlm_temp_o = alm_copypl(hp.read_alm(f"/users/odarwish/scratch/JOINTRECONSTRUCTION/apo_new/simswalpha/sim_{originalidx:04d}_olm.fits"), lmax = lmax_qlm)#NOTE
    alpha_lm = alm_copypl(hp.read_alm(f"/users/odarwish/scratch/JOINTRECONSTRUCTION/apo_new/simswalpha/sim_{originalidx:04d}_alpha_lm.fits"), lmax = lmax_qlm)#NOTE

    #get phases X
    phases_x = fg_phases(dlm_temp, simidx+3)
    dlm = np.concatenate([x*phases_x for x in [alpha_lm, dlm_temp, dlm_temp_o]], axis = 0) #original fields with phases x
    set_operator(itlib, dlm)
    eblm_dat_i = itlib.filter.synalm(cls_unl, seed = simidx)
    
    dlm = np.concatenate([x*phases_x for x in np.split(dlmsolution, number_of_operators)], axis = 0) #original solution with phases x
    set_operator(itlib, dlm)
    itlib_filter_sol_x = copy.deepcopy(itlib.filter) #COPY itlib X
    mchain = multigrid.multigrid_chain(itlib.opfilt, itlib.chain_descr, itlib.cls_filt, itlib.filter)
    elm_wf_i = np.zeros(Alm.getsize(itlib.filter.lmax_sol, itlib.filter.mmax_sol), dtype=complex)
    mchain.solve(elm_wf_i, eblm_dat_i, dot_op=itlib.filter.dot_op())

    #get phases Y
    phases_y = fg_phases(dlm_temp, simidx+10)
    dlm = np.concatenate([x*phases_y for x in [alpha_lm, dlm_temp, dlm_temp_o]], axis = 0) #original fields with phases y
    set_operator(itlib, dlm)    
    eblm_dat_j = itlib.filter.synalm(cls_unl, seed = simidx+5)

    dlm = np.concatenate([x*phases_y for x in np.split(dlmsolution, number_of_operators)], axis = 0) #original solution with phases y
    set_operator(itlib, dlm)
    itlib_filter_sol_y = copy.deepcopy(itlib.filter) #COPY itlib Y
    mchain = multigrid.multigrid_chain(itlib.opfilt, itlib.chain_descr, itlib.cls_filt, itlib.filter)
    elm_wf_j = np.zeros(Alm.getsize(itlib.filter.lmax_sol, itlib.filter.mmax_sol), dtype=complex)
    mchain.solve(elm_wf_j, eblm_dat_j, dot_op=itlib.filter.dot_op())

    G_total = []
    G_total_ii = []
    G_total_jj = []
    G_total_j_i = []
    G_total_di = []
    G_total_id = []

    for o in itlib.filter.operators:
        which = o.name
        if which == 'p':

            G, C = itlib_filter_sol_x.get_qlms(eblm_dat_i, elm_wf_i, q_geom, elm_wf_j, which, filter_leg2 = itlib_filter_sol_y) #one leg and another leg
            hp_almxfl(G, itlib.filter._h2p(which, lmax_qlm), mmax_qlm, True)
            hp_almxfl(C, itlib.filter._h2p(which, lmax_qlm), mmax_qlm, True)
            G_total.append(G) #ij

            if "o" in itlib.filter.operators.names:
                G_total.append(C)
            ########################################### 


            G, C = itlib_filter_sol_x.get_qlms(eblm_dat_i, elm_wf_i, q_geom, wf_data, which, filter_leg2 = itlib_filter_sol_d) #one leg and another leg
            hp_almxfl(G, itlib.filter._h2p(which, lmax_qlm), mmax_qlm, True)
            hp_almxfl(C, itlib.filter._h2p(which, lmax_qlm), mmax_qlm, True)
            G_total_id.append(G)

            if "o" in itlib.filter.operators.names:
                G_total_id.append(C)
            ########################################### 

            G, C = itlib_filter_sol_d.get_qlms(data, wf_data, q_geom, elm_wf_i, which, filter_leg2 = itlib_filter_sol_x) #one leg and another leg
            hp_almxfl(G, itlib.filter._h2p(which, lmax_qlm), mmax_qlm, True)
            hp_almxfl(C, itlib.filter._h2p(which, lmax_qlm), mmax_qlm, True)
            G_total_di.append(G)

            if "o" in itlib.filter.operators.names:
                G_total_di.append(C)
            ########################################### 

            G, C = itlib_filter_sol_y.get_qlms(eblm_dat_j, elm_wf_j, q_geom, elm_wf_i, which, filter_leg2 = itlib_filter_sol_x) #one leg and another leg
            hp_almxfl(G, itlib.filter._h2p(which, lmax_qlm), mmax_qlm, True)
            hp_almxfl(C, itlib.filter._h2p(which, lmax_qlm), mmax_qlm, True)
            G_total_j_i.append(G)

            if "o" in itlib.filter.operators.names:
                G_total_j_i.append(C)
            ###########################################

            G, C = itlib_filter_sol_x.get_qlms(eblm_dat_i, elm_wf_i, q_geom, which = which) #one leg and another leg
            hp_almxfl(G, itlib.filter._h2p(which, lmax_qlm), mmax_qlm, True)
            hp_almxfl(C, itlib.filter._h2p(which, lmax_qlm), mmax_qlm, True)
            G_total_ii.append(G)

            if "o" in itlib.filter.operators.names:
                G_total_ii.append(C)
            ########################################### 

            G, C = itlib_filter_sol_y.get_qlms(eblm_dat_j, elm_wf_j, q_geom, which = which) #one leg and another leg
            hp_almxfl(G, itlib.filter._h2p(which, lmax_qlm), mmax_qlm, True)
            hp_almxfl(C, itlib.filter._h2p(which, lmax_qlm), mmax_qlm, True)
            G_total_jj.append(G)

            if "o" in itlib.filter.operators.names:
                G_total_jj.append(C)
            ########################################### 


        elif which == 'o':
            if "p" not in itlib.filter.operators.names:
                G, C = itlib.filter.get_qlms(eblm_dat_i, elm_wf_i, q_geom, elm_wf_j, which)
                hp_almxfl(C, itlib.filter._h2p(which, lmax_qlm), mmax_qlm, True)
            else:
                print("Skip gradient o, already added.")
        else:
            G = itlib_filter_sol_x.get_qlms(eblm_dat_i, elm_wf_i, q_geom, elm_wf_j, which, filter_leg2 = itlib_filter_sol_y)
            G_total.append(G)
            ########################################### 
            G = itlib_filter_sol_d.get_qlms(data, wf_data, q_geom, elm_wf_i, which, filter_leg2 = itlib_filter_sol_x)
            G_total_di.append(G_total_data)
            ########################################### 
            G = itlib_filter_sol_x.get_qlms(eblm_dat_i, elm_wf_i, q_geom, wf_data, which, filter_leg2 = itlib_filter_sol_d)
            G_total_id.append(G)
            ########################################### 
            G = itlib_filter_sol_y.get_qlms(eblm_dat_i, elm_wf_i, q_geom, elm_wf_j, which, filter_leg2 = itlib_filter_sol_x)
            G_total_j_i.append(G)
            ########################################### 
            G = itlib_filter_sol_x.get_qlms(eblm_dat_i, elm_wf_i, q_geom, which = which)
            G_total_ii.append(G)
            ########################################### 
            G = itlib_filter_sol_y.get_qlms(eblm_dat_j, elm_wf_j, q_geom, which = which)
            G_total_jj.append(G)
            ###########################################


    G_total = np.concatenate(G_total, axis = 0)
    G_total_j_i = np.concatenate(G_total_j_i, axis = 0)
    G_total_ii = np.concatenate(G_total_ii, axis = 0)
    G_total_jj = np.concatenate(G_total_jj, axis = 0)

    G_total_di = np.concatenate(G_total_di, axis = 0)
    G_total_id = np.concatenate(G_total_id, axis = 0)

    cacher.cache(fn_lik(simidx, x, y), np.array(G_total))
    cacher.cache(fn_lik(simidx, y, x), np.array(G_total_j_i))
    cacher.cache(fn_lik(simidx, x, x), np.array(G_total_ii))
    cacher.cache(fn_lik(simidx, y, y), np.array(G_total_jj))
    cacher.cache(fn_lik(simidx, "d", x), np.array(G_total_di))
    cacher.cache(fn_lik(simidx, x, "d"), np.array(G_total_id))
   
    return 0

def load_calculated_mf(itlib, itr, mcs, key:str = 'p', zerolensing = False):
    fn_lik = lambda this_idx : f'{itlib.h}lm_grad{key.lower()}det_it{itr:03d}_sim{this_idx:04d}' + '_nolensing'*zerolensing


    cacher = itlib.cacher

    all_present = all(os.path.exists(opj(cacher.lib_dir, fn_lik(idx)+".npy")) for idx in np.unique(mcs))
    print("ALL PRESENT", all_present)

    assert all_present
    
    if not all_present:
        return 0
    

    _Gmfs = []
    for idx in np.unique(mcs):
        _Gmfs.append(np.concatenate(cacher.load(fn_lik(idx)), axis = 0))

    mean = np.mean(_Gmfs, axis=0)
    return mean



def get_graddet_sim_mf_trick(itlib:cs_iterator.qlm_iterator, itr:int, mcs:np.ndarray, 
                             key:str='p', mf_phas:phas.lib_phas=None, zerolensing:bool=False, recache=False):   
    """Buid the gradient MF using the trick of Carron and Lewis 2017 Appendix B
    
        Args:
            itlib: iterator instance to compute the ds ans ss for
            itr: iteration index of MAP phi
            mcs: sim indices
            key: QE key
            mf_phase: phases of the alm for the simulations
            zerolensing: Set the lensing field to zero in the gradient
    """
    #FIXME: This script can easily be paralelized with MPI in the for loops

    fn_lik = lambda this_idx : f'{itlib.h}lm_grad{key.lower()}det_it{itr:03d}_sim{this_idx:04d}' + '_nolensing'*zerolensing
            
    cacher = itlib.cacher
    
    mf_key=1 # Uses the trikc of Carron and Lewis 2017

    dlm = itlib.get_hlm(itr - 1, key)

    number_of_operators = len(itlib.filter.operators)

    for opindex, value in enumerate(np.split(dlm, number_of_operators)):
        which = itlib.filter.operators[opindex].name
        print("Getting", which)
        if which == "p":
            dlm_temp = value.copy()
            itlib.hlm2dlm(dlm_temp, True)

            dlm_temp_o = None

            if  "o" in itlib.filter.operators.names:
                oindex = itlib.filter.operators.names.index("o")
                dlm_temp_o = np.split(dlm, number_of_operators)[oindex]
                itlib.hlm2dlm(dlm_temp_o, True) 

            ffi = itlib.filter.operators.get("p").field.change_dlm([dlm_temp, dlm_temp_o], itlib.mmax_qlm, cachers.cacher_mem(safe=False))
            itlib.filter.operators.set_field(ffi, which = "p")

        elif (which == "o"):
            if ("p" not in itlib.filter.operators.names):
                dlm_temp = value.copy()
                itlib.hlm2dlm(dlm_temp, True)
                ffi = itlib.filter.operators.get("o").field.change_dlm([None, dlm_temp], itlib.mmax_qlm, cachers.cacher_mem(safe=False))
                itlib.filter.operators.set_field(ffi, which = "o")
            else:
                print("Skipping this.")

        elif which == "a":
            alpha_map = itlib.filter.ninv_geom.synthesis(dlm.copy(), spin = 0, lmax = itlib.lmax_qlm, mmax = itlib.mmax_qlm, nthreads = 128).squeeze()
            itlib.filter.operators.set_field(alpha_map, which = "a")
        elif which == "f":
            tau_map = itlib.filter.ninv_geom.synthesis(dlm.copy(), spin = 0, lmax = itlib.lmax_qlm, mmax = itlib.mmax_qlm, nthreads = 128).squeeze()
            itlib.filter.operators.set_field(tau_map, which = "f")
        else:
            raise ValueError

    mchain = multigrid.multigrid_chain(itlib.opfilt, itlib.chain_descr, itlib.cls_filt, itlib.filter)


    q_geom = pbdGeometry(itlib.k_geom, pbounds(0., 2 * np.pi))
    #q_geom = itlib.filter.ffi.pbgeom

    _Gmfs = []
    for idx in np.unique(mcs):
        if not cacher.is_cached(fn_lik(idx)) or recache:
            print(f'Doing MF sim {idx}' + ' no lensing'*zerolensing)
            if mf_phas is not None:
                phas_x = mf_phas.get_sim(idx, idf=0)
                phas_y = mf_phas.get_sim(idx, idf=1)
                phas = np.array([phas_x, phas_y])
                # phas = alm_copy(phas, None, itlib.filter.lmax_len, itlib.filter.mmax_len) 
            else:
                phas = None

            t0 = time.time()
            G = itlib.filter.get_qlms_mf(key, mf_key, q_geom, mchain, cls_filt=itlib.cls_filt, phas=phas, lmax_qlm = itlib.lmax_qlm, mmax_qlm = itlib.mmax_qlm)
            #hp_almxfl(G, itlib._h2p(itlib.lmax_qlm), itlib.mmax_qlm, True)
            print('get_qlm_mf calculation done; (%.0f secs)' % (time.time() - t0))

            itlib.cacher.cache(fn_lik(idx), G)
        
        _Gmfs.append(itlib.cacher.load(fn_lik(idx)))
    print("Length of Gmfs", len(_Gmfs))
    return np.mean(_Gmfs, axis = 0)

def get_graddet_sim_mf_true(qe_key:str, itr:int, mcs:np.ndarray, itlib:cs_iterator.qlm_iterator, 
                            cmb_phase:phas.lib_phas, 
                            noise_phase:phas.lib_phas, 
                            assert_phases_exist=False, zerolensing=False, recache=False):
    """Builds grad MF from averaging sims with lensing field equal to MAP field

        Args:
            qe_key: 'p_p' for Pol-only, 'ptt' for T-only, 'p_eb' for EB-only, etc
            itr: iteration index of MAP phi
            mcs: sim indices
            itlib: iterator instance to compute the gradient for
            cmb_phas: igenerates the unlensed CMB phases for the sims
            noise_phase: phase for the noise of the CMB
            assert_phases_exist: set this if you expect the phases to be already calculatex
            zerolensing: Set the lensing field to zero in the sims and in the gradient 
    """
    #FIXME: This script can easily be paralelized with MPI in the for loops
    
    #assert qe_key == 'ptt', 'Phases not implemented for pol'
    assert hasattr(itlib.filter, 'synalm')
    
    # Setting up fitlering instance:
    dlm = itlib.get_hlm(itr - 1, 'p')

    number_of_operators = len(itlib.filter.operators)

    for opindex, value in enumerate(np.split(dlm, number_of_operators)):
        which = itlib.filter.operators[opindex].name
        print("Getting", which)
        if which == "p":
            dlm_temp = value.copy()
            itlib.hlm2dlm(dlm_temp, True)

            dlm_temp_o = None

            if  "o" in itlib.filter.operators.names:
                oindex = itlib.filter.operators.names.index("o")
                dlm_temp_o = np.split(dlm, number_of_operators)[oindex]
                itlib.hlm2dlm(dlm_temp_o, True) 

            ffi = itlib.filter.operators.get("p").field.change_dlm([dlm_temp, dlm_temp_o], itlib.mmax_qlm, cachers.cacher_mem(safe=False))
            itlib.filter.operators.set_field(ffi, which = "p")

        elif (which == "o"):
            if ("p" not in itlib.filter.operators.names):
                dlm_temp = value.copy()
                itlib.hlm2dlm(dlm_temp, True)
                ffi = itlib.filter.operators.get("o").field.change_dlm([None, dlm_temp], itlib.mmax_qlm, cachers.cacher_mem(safe=False))
                itlib.filter.operators.set_field(ffi, which = "o")
            else:
                print("Skipping this.")

        elif which == "a":
            alpha_map = itlib.filter.ninv_geom.synthesis(dlm.copy(), spin = 0, lmax = itlib.lmax_qlm, mmax = itlib.mmax_qlm, nthreads = 128).squeeze()
            itlib.filter.operators.set_field(alpha_map, which = "a")
        elif which == "f":
            tau_map = itlib.filter.ninv_geom.synthesis(dlm.copy(), spin = 0, lmax = itlib.lmax_qlm, mmax = itlib.mmax_qlm, nthreads = 128).squeeze()
            itlib.filter.operators.set_field(tau_map, which = "f")
        else:
            raise ValueError


    chain_descr = itlib.chain_descr
    mchain = multigrid.multigrid_chain(itlib.opfilt, chain_descr, itlib.cls_filt, itlib.filter)
    q_geom = pbdGeometry(itlib.k_geom, pbounds(0., 2 * np.pi))
 
    ivf_cacher = cachers.cacher_npy(opj(itlib.lib_dir, f'mf_sims_itr{itr:03d}'))
    print(ivf_cacher.lib_dir)
    
    fn_wf = lambda this_idx : 'dat_wf_filtersim_%04d'%this_idx + '_nolensing'*zerolensing # Wiener-filtered sim
    fn = lambda this_idx : 'dat_filtersim_%04d'%this_idx + '_nolensing'*zerolensing # full sims
    fn_unl = lambda this_idx : 'unllm_filtersim_%04d'%this_idx # Unlensed CMB to potentially share between parfile
    fn_qlm = lambda this_idx : 'qlm_mf_sim_%04d'%this_idx + '_nolensing'*zerolensing # qlms sim 
    
        
    for i in np.unique(mcs):
        idx = int(i)
        if not ivf_cacher.is_cached(fn_wf(idx)) or not ivf_cacher.is_cached(fn(idx)) or recache:
            print(f'MF grad getting WF sim {idx}')

            xlm_dat = itlib.filter.synalm(itlib.cls_filt, cmb_phas=cmb_phase.get_sim(idx, idf=0), noise_phase=noise_phase.get_sim(idx, idf=0))
            assert hp.Alm.getlmax(xlm_dat.size, itlib.filter.mmax_len) == itlib.filter.lmax_len, (hp.Alm.getlmax(xlm_dat.size, itlib.filter.mmax_len),  itlib.filter.lmax_len)
            
            ivf_cacher.cache(fn(idx), xlm_dat)
            # Get the WF CMB map
            soltn = np.zeros(uhp.Alm.getsize(itlib.lmax_filt, itlib.mmax_filt), dtype=complex)
            mchain.solve(soltn, ivf_cacher.load(fn(idx)), dot_op=itlib.filter.dot_op())
            ivf_cacher.cache(fn_wf(idx), soltn)

    if qe_key == 'p_p':
        get_qlms = itlib.filter.get_qlms
    elif qe_key == 'ptt':
        get_qlms = itlib.filter.get_qlms

    q_geom = itlib.filter.ffi.pbgeom  
    #FIXME Seems like it is different from pbdGeometry(itlib.k_geom, pbounds(0., 2 * np.pi))
    # but should be the same, maybe only the memory adress is different so python see it as different in itlib.filtr._get_gpmap

    # Get the QEs gradients
    _qlms = []
    for idx in np.unique(mcs):
        if not ivf_cacher.is_cached(fn_qlm(idx)):
            wf_i = ivf_cacher.load(fn_wf(idx))
            qlm = get_qlms(ivf_cacher.load(fn(idx)), wf_i, q_geom)[0]
            ivf_cacher.cache(fn_qlm(idx), qlm)
        _qlms.append(ivf_cacher.load(fn_qlm(idx)))
    return np.array(_qlms)