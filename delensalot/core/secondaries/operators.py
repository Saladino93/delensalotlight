import numpy as np
import inspect
from typing import List



def fg_phases(mappa: np.ndarray, seed: int = 0):
    np.random.seed(seed)
    f = lambda z: np.exp(1j*np.random.uniform(0., 2.*np.pi, size = z.shape))
    return f(mappa)

def randomize(filtr, x, idx = 0, shift = 100, spin = 2):
    if shift == 0:
        return x
    else:
        if np.iscomplexobj(x):
            return x*fg_phases(x[0] if x.ndim == 2 else x, idx+shift) #assumes 1 or 2 dimensions max
        else:
            #print("Randomizer for real maps")
            xlm = filtr.ninv_geom.adjoint_synthesis(x, spin, filtr.mmax_len, filtr.lmax_len, filtr.operators.sht_tr)
            xlm_rand = xlm*fg_phases(xlm if xlm.ndim == 2 else xlm, idx+shift)
            #print(fg_phases(xlm[0], idx+shift))
            return filtr.ninv_geom.synthesis(xlm_rand, spin, filtr.mmax_len, filtr.lmax_len, filtr.operators.sht_tr)

class Operators():
    def __init__(self, operators_order = [], ignore_calling: List[str] = []):
        self.operators_order = operators_order
        self._names = [op.name for op in self.operators_order]
        self.sht_tr = min([op.sht_tr for op in self.operators_order])
        assert set(ignore_calling) <= set(self._names), f"ignore_calling must be a subset of {self._names}"
        self.ignore_calling = ignore_calling
        self.not_ignored = [op.name for op in self.operators_order if op.name not in self.ignore_calling]


    def get(self, which = ""):
        for op in self.operators_order:
            if op.name == which:
                return op
        return None

    def apply_operators(self, elbm, which = "", backwards = False, ignore = [], **kwargs):
        lista = self.operators_order if not backwards else self.operators_order[::-1]
        lista_not_ignored_names = self.not_ignored if not backwards else self.not_ignored[::-1]
        #remove which from ignored
        
        for op in lista:
            if (op.name in ignore) or (op.name in self.ignore_calling):
                continue
            signature = inspect.signature(op.__call__)
            parameters = signature.parameters
            filtered_kwargs = {key: kwargs[key] for key in parameters if key in kwargs}
            out_real = (op.name == lista_not_ignored_names[-1]) and (which != "")
            elbm = op(elbm, derivative = (op.name == which), backwards = backwards, out_real = out_real, **filtered_kwargs)
        
        return elbm
    
    def __call__(self, eblm, which = "", backwards = False, ignore = [], **kwargs):
        return self.apply_operators(eblm, which = which, backwards = backwards, ignore = ignore, **kwargs)
    
    def set_field(self, field, which = ""):
        for op in self.operators_order:
            if op.name == which:
                op.set_field(field)
                break

    @property
    def names(self):
        return self._names

    def __len__(self):
        return len(self.operators_order)

    def __getitem__(self, index):
        return self.operators_order[index]

    def __iter__(self):
        return iter(self.operators_order)
    
    def lmax(self, which = ""):
        for op in self.operators_order:
            if op.name == which:
                return op.lmax
        return 0

    def mmax(self, which = ""):
        for op in self.operators_order:
            if op.name == which:
                return op.mmax
        return 0


class Operator():
    def __init__(self, name: str, lmax: int, mmax: int, sht_tr: int, disable: bool = False):
        assert name != "", "Operator name cannot be empty"
        self.name = name     
        self.lmax = lmax
        self.mmax = mmax
        self.sht_tr = sht_tr
        self.disable = disable

    def set_field(self, field):
        pass

    @property
    def field(self):
        pass



class AmplitudeModulation(Operator):

    def __init__(self, name: str, lmax: int, mmax: int, sht_tr: int, disable: bool = False):
        super().__init__(name, lmax, mmax, sht_tr, disable)
        self.amplitude = None

    def __call__(self, eblm, backwards = False, derivative = False, spin = None, q_pbgeom = None, 
                 lmax_in = None, lmax_out = None, mmax_out = None, out_sht_mode = "STANDARD", apply_weights = True, out_real = False):
        
        assert q_pbgeom is not None, "q_pbgeom cannot be None"

        pass

    def set_amplitude(self, amplitude):
        pass

    def set_field(self, field):
        self.set_amplitude(field)

    @property
    def field(self):
        return self.amplitude

    def amplitude(self, QU, backwards=False):
        if self.amplitude is not None:
            return self._multiply(QU, self.amplitude if not backwards else self.amplitude)
        else:
            return QU

    def derivative_amplitude(self, QU):
        pass
    
    @staticmethod
    def _multiply(a, b):
        return np.einsum('ab, b -> ab', a, b)
    

    def get_qlms(self, filtr, eblm_dat: np.ndarray or list, elm_wf: np.ndarray, q_pbgeom, alm_wf_leg2:None or np.ndarray =None, which = "f", shift_1: int = 0, shift_2: int = 0, mean_field = False, filter_leg2 = None, cache = False):
        pass

