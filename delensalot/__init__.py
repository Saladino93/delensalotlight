import os, sys
from os.path import join as opj
from pathlib import Path
import hashlib, psutil, shutil
import numpy as np
import healpy as hp

if "SCRATCH" not in os.environ:
    if 'site-pack' not in os.path.dirname(__file__):
        os.environ["SCRATCH"] = opj(Path(__file__).parent.parent, "reconstruction/")
    else:
        # If delensalot is installed without dev mode, put SCRATCH to user folder.
        os.environ["SCRATCH"] = os.path.expanduser('~/reconstruction')
    if not os.path.exists(os.environ["SCRATCH"]):
        os.makedirs(os.environ["SCRATCH"])