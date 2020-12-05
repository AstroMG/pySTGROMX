import numpy as np
import os
import h5py
from scipy.interpolate import splrep, splev
from scipy.interpolate import RectBivariateSpline, bisplev

def load_dict_from_hdf5(filename, mode='r'):

    if mode not in ['r', 'r+', 'a+']:
        raise Exception('>>> read mode error')
    with h5py.File(filename, mode) as h5file:
        return __recursively_load_dict_contents_from_group(h5file, '/')

def __recursively_load_dict_contents_from_group(h5file, path):

    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = __recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

class FunctionWrapper(object):
    '''
    This is a hack to make a function pickleable when ``args`` or ``kwargs`` are also included.

    '''

    def __init__(self, f, *args, **kwargs):
        #print('funw init')
        self.f = f
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

    def __call__(self, x):
        return self.f(x, *self.args, **self.kwargs)

class StarFunctionWrapper(FunctionWrapper):
    def __call__(self, x):
        return self.f(*x, *self.args, **self.kwargs)

def spline2d(x, y, xdeg=5, ydeg=5, s=0):
    # args = np.argsort(x)
    assert len(x) == 2, "Sample data not 2-dimensional."

    dim_y = len(np.shape(y))
    if dim_y == 2:
        spline = RectBivariateSpline(x[0], x[1], y, kx=xdeg, ky=ydeg, s=s)
        _eval = list(spline.tck)

        ans = StarFunctionWrapper(bisplev,[_eval[0], _eval[1], _eval[2], xdeg, ydeg], dx=0, dy=0)

        #def ans(X, dx=0, dy=0):
        #    return bisplev(X[0], X[1], [_eval[0], _eval[1], _eval[2], xdeg, ydeg], dx=dx, dy=dy)

    elif dim_y > 2:
        _eval = []
        for yy in y:
            spline = RectBivariateSpline(x[0], x[1], yy, kx=xdeg, ky=ydeg, s=s)
            _eval.append(list(spline.tck))

        ans = np.array([StarFunctionWrapper(bisplev, [ee[0], ee[1], ee[2], xdeg, ydeg], dx=0, dy=0) for ee in _eval])

        #def ans(X, dx=0, dy=0):
        #    return np.array([bisplev(X[0], X[1], [ee[0], ee[1], ee[2], xdeg, ydeg], dx=dx, dy=dy) for ee in _eval])

    else:
        raise Exception("Unexpected dimension for y data array(s).")

    return ans, _eval
    
def MultivariateFits(x, y, xdeg=5, ydeg=5, s=0):
    eval, _eval = spline2d(x, y, xdeg=xdeg, ydeg=ydeg, s=s)
    return eval

class LoadModel:
    def __init__(self, EOS_name, label='', path='.'):
        #filename = EOS_name + '_mod.hdf5'
        file_path = os.path.abspath(path + '/' + label + '/' + EOS_name + '_mod.hdf5')
        filename = file_path
        info = load_dict_from_hdf5(filename)
        self.logAlphas = info['logAlpha0']
        self.betas = info['beta0']
        self.e_cs = info['ec_nodes']
        self.eim_data = {'mA':[], 'R':[], 'logAlphaA':[], 'logIA':[], 'logbetaA':[], 'logkA':[]}
        self.eim_indices = {'mA':[], 'R':[], 'logAlphaA':[], 'logIA':[], 'logbetaA':[], 'logkA':[]}
        self.eim_B = {'mA':[], 'R':[], 'logAlphaA':[], 'logIA':[], 'logbetaA':[], 'logkA':[]}
        self.rb_basis = {'mA':[], 'R':[], 'logAlphaA':[], 'logIA':[], 'logbetaA':[], 'logkA':[]}
        self.rb_errors = {'mA':[], 'R':[], 'logAlphaA':[], 'logIA':[], 'logbetaA':[], 'logkA':[]}

        self.fits = {'mA':[], 'R':[], 'logAlphaA':[], 'logIA':[], 'logbetaA':[], 'logkA':[]}
        for key in ['mA', 'R', 'logAlphaA', 'logIA', 'logbetaA', 'logkA']:
            tmp = info[key]
            self.eim_data[key] = tmp['eim']['data']
            self.eim_indices[key] = tmp['eim']['indices']
            self.eim_B[key] = tmp['eim']['B']
            self.rb_basis[key] = tmp['rb']['basis']
            self.rb_errors[key] = tmp['rb']['errors']

            self.fits[key] = []
            for dd in self.eim_data[key]:
                fit = MultivariateFits([self.logAlphas, self.betas], 
                    dd.reshape(len(self.logAlphas), len(self.betas)),
                    xdeg=5, ydeg=5, s=0)
                self.fits[key].append(fit)
    
    def _evo(self, key, pool, val_ec):
        fits_tmp = self.fits[key]
        fit_evals = np.array([fits_tmp[ii](pool) 
                for ii in range(len(self.eim_indices[key]))])
        values =  np.dot(fit_evals, self.eim_B[key])
        bspl =  splrep(self.e_cs, values)
        return splev(val_ec, bspl)

    def __call__(self, logAlpha, beta, e_c, keys=['mA','R','AlphaA','IA','betaA','kA'], s=0):
        pool = [[logAlpha], [beta]]
        if logAlpha < self.logAlphas[0] or logAlpha > self.logAlphas[-1] \
                    or beta < self.betas[0] or beta > self.betas[-1]:
            raise Exception("Range error with logAlpha or beta")
        
        vals=()
        for key in keys:
            if(key in ['mA','R']):
                val = self._evo(key=key, pool=pool, val_ec=e_c)
            elif(key in ['alphaA']):
                val = np.exp(self._evo(key='logAlphaA', pool=pool, val_ec=e_c))
            else:
                val = np.exp(self._evo(key='log'+key, pool=pool, val_ec=e_c))
            vals += (val,)
        return vals