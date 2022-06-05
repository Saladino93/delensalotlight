#!/usr/bin/env python

"""handler.py: Base handler for lensing reconstruction pipelines.
    Collects lerepi jobs defined in DLENSALOT_jobs
    Extracts models needed for each job via x2y_Transformer
    runs all jobs
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"


import os
import sys
import importlib.util as iu
import shutil

import logging
from logdecorator import log_on_start, log_on_end

from plancklens.helpers import mpi

from lerepi.core.visitor import transform
from lerepi.core.transformer.param2dlensalot import p2j_Transformer, p2T_Transformer, transform


class handler():

    def __init__(self, parser):
        self.paramfile = handler.load_paramfile(parser.config_file, 'paramfile')
        TEMP = transform(self.paramfile.dlensalot_model, p2T_Transformer())
        self.store(parser, self.paramfile, TEMP)


    @log_on_start(logging.INFO, "Start of collect_jobs()")
    @log_on_end(logging.INFO, "Finished collect_jobs()")
    def collect_jobs(self):
        self.jobs = transform(self.paramfile.dlensalot_model, p2j_Transformer())


    @log_on_start(logging.INFO, "Start of get_jobs()")
    @log_on_end(logging.INFO, "Finished get_jobs()")
    def get_jobs(self):
        return self.jobs


    @log_on_start(logging.INFO, "Start of get_jobs()")
    @log_on_end(logging.INFO, "Finished get_jobs()")
    def init_job(self, job):
        model = transform(*job[0])
        j = job[1](model)
        j.collect_jobs()
        return j


    @log_on_start(logging.INFO, "Start of run()")
    @log_on_end(logging.INFO, "Finished run()")
    def run(self):
        for transf, job in self.jobs:
            model = transform(*transf)           
            j = job(model)
            j.collect_jobs()
            j.run()


    @log_on_start(logging.INFO, "Start of store()")
    @log_on_end(logging.INFO, "Finished store()")
    def store(self, parser, paramfile, TEMP):
        """ Store the dlensalot_model as parameterfile in TEMP, to use if run resumed

        Args:
            parser (_type_): _description_
            paramfile (_type_): _description_
            TEMP (_type_): _description_
        """
        dostore = False
        if parser.resume == '':
            dostore = True
            # This is only done if not resuming. Otherwise file would already exist
            if os.path.isfile(parser.config_file) and parser.config_file.endswith('.py'):
                # if the file already exists, check if something changed
                if os.path.isfile(TEMP+'/'+parser.config_file.split('/')[-1]):
                    dostore = False
                    logging.warning('Param file {} already exist. Checking differences.'.format(TEMP+'/'+parser.config_file.split('/')[-1]))
                    paramfile_old = handler.load_paramfile(TEMP+'/'+parser.config_file.split('/')[-1], 'paramfile_old')   
                    for key, val in paramfile_old.dlensalot_model.__dict__.items():
                        for k, v in val.__dict__.items():
                            if v != paramfile.dlensalot_model.__dict__[key].__dict__[k]:
                                if callable(v):
                                    # If it's a function, we can test if bytecode is the same as a simple check won't work due to pointing to memory location
                                    if v.__code__.co_code != paramfile.dlensalot_model.__dict__[key].__dict__[k].__code__.co_code:
                                        logging.error("{} changed. Attribute {} had {} before, it's {} now.".format(key, k, v, paramfile.dlensalot_model.__dict__[key].__dict__[k]))
                                        logging.error('Exit. Check config file.')
                                        sys.exit()
                    logging.info('Param file look the same. Resuming where I left off last time.')

        if dostore:
            if mpi.rank == 0:
                if not os.path.exists(TEMP):
                    os.makedirs(TEMP)
                shutil.copyfile(parser.config_file, TEMP+'/'+parser.config_file.split('/')[-1])
            logging.info('Parameterfile stored at '+ TEMP+'/'+parser.config_file.split('/')[-1])
        else:
            if parser.resume == '':
                # Only give this info when not resuming
                logging.info('Matching parameterfile found. Resuming where I left off.')


    @log_on_start(logging.INFO, "Start of load_paramfile()")
    @log_on_end(logging.INFO, "Finished load_paramfile()")
    def load_paramfile(directory, descriptor):
        """Load parameterfile

        Args:
            directory (_type_): _description_
            descriptor (_type_): _description_

        Returns:
            _type_: _description_
        """
        spec = iu.spec_from_file_location('paramfile', directory)
        p = iu.module_from_spec(spec)
        sys.modules[descriptor] = p
        spec.loader.exec_module(p)
        return p
