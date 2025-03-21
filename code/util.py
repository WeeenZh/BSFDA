from scipy.special import digamma, loggamma
import logging
import numpy as np
import matplotlib.pylab as plt
import pprint
import copy
from sklearn.cluster import KMeans
import math
import numexpr as ne

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Kernel, Hyperparameter


EPS = 10**(np.ceil(np.log10(np.finfo(float).eps)/2)*2 + 2)


def get_kernel_list_and_coef_func_index(
    x_up_list,
    y_up_list,
    dir_debug,
    n_cluster_kernel = 5,
    normalize_rbf = True,
    kernel_coef = 1,
    ):
    '''
    get kernel list and coef_func_index
    ---
    x_up_list:
        list of x_up, x_up.shape = (1, n_up)
    y_up_list:
        list of y_up, y_up.shape = (1, n_up)
    n_cluster_kernel:
        number of clusters for kernel length scale
    dir_debug:
        directory for debug
    normalize_rbf:
        whether to normalize the rbf kernel
    kernel_coef:
        kernel coefficient
    ---
    kernel_list:
        list of kernels
    coef_func_index:
        coefficient for the size of index set for basis functions. To reduce computation with similar basis functions, the original index set is clustered into a smaller set. The size of the smaller set is coef_func_index times the largest number of measurements of sample functions. coef_func_index is calculated based on the length scale of the kernel and the length of the domain of the data.
    '''
    # NOTE the basis functions should have the same scale as the data, this could be replaced by the mean of the data
    multipier_constant_value= function_sample_scale_percentile(function_sample_list= y_up_list, q=99) * kernel_coef
    logging.critical(f'multipier_constant_value= {multipier_constant_value}')

    x_min = np.min([np.min(x) for x in x_up_list])
    x_max = np.max([np.max(x) for x in x_up_list])
    x_domain_lengxh = x_max - x_min


    ls_list = cv_kernel_kmeans(
        length_scale_list= [
            x_domain_lengxh * 2**( 4 - c/2)
            for c in range(17)
        ],
        n_clusters=n_cluster_kernel,
        x_up_list= x_up_list,
        y_up_list= y_up_list,
        path_fig=f'{dir_debug}/length_scale_k_means.pdf',
        multipier_constant_value=multipier_constant_value,
        normalize_rbf= normalize_rbf,
        x_domain_length= x_domain_lengxh,
    )
    kernel_list= [
        RBFKernel2(sigma=0, length_scale= c, multiplier= multipier_constant_value/rbf_integral(length_scale=c,a=x_domain_lengxh/2) if normalize_rbf else multipier_constant_value )
        for i,c in enumerate(ls_list)
    ]
    logging.critical(f'choose kernel ls_list= {ls_list}')


    # estimate the coefficient for the size of the index set
    ratio_n_index_length_scale = 0.1
    n_index_expect = x_domain_lengxh / ( np.min(ls_list) * ratio_n_index_length_scale )
    coef_func_index=  n_index_expect / np.max([z.shape[1] for z in x_up_list])
    logging.info(f'coef_func_index= {coef_func_index}')

    return kernel_list, coef_func_index


def loglike_gamma(
    x,
    a,
    b,
    ):
    '''
    log likelihood of gamma distribution
    ---
    x: 
        scalar, data
    a: 
        scalar, shape parameter
    b: 
        scalar, rate parameter
    '''
    loglike = a * np.log(b) - loggamma(a) + (a-1) * np.log(x) - b * x
    return loglike



def loglike_normal(
    x,
    mu = None,
    var_log = None,
    cov = None,
    cov_is_diagonal = False,
    ):
    '''
    log likelihood of multivariate normal distribution
    ---
    x: 
        (n, d), data
    mu: 
        (1, d), mean of multivariate normal distribution, default to 0
    var_log:
        scalar, log variance of covariance matrix if it is a scalar multiple of identity matrix
    cov: 
        (d, d)
    cov_is_diagonal:
        bool, whether covariance matrix is diagonal
    '''

    assert ((cov is not None) or (var_log is not None)), 'either cov or var_log must be specified'

    n, d = x.shape
    
    if mu is None:
        x_tilde = x
    else:
        x_tilde = x - mu

    if var_log is not None:
        if var_log == 0:
            cov_logdet = 0
            mhlnbs = ( x_tilde**2 ).sum()
        else:
            cov_logdet = d * var_log
            mhlnbs = ( x_tilde**2 / np.exp(var_log) ).sum()
    else:
        if cov_is_diagonal:
            cov_diag = np.diag( cov )
            cov_logdet = np.log( cov_diag ).sum()
            mhlnbs = ( x_tilde**2 / cov_diag[None,:] ).sum()
        else:
            # use np.linalg.slogdet instead of np.log(np.linalg.det) to avoid overflow
            sign, cov_logdet = np.linalg.slogdet( cov )
            # sign has to be 1
            assert sign == 1, 'covariance matrix is not positive definite'
            mhlnbs = ( x_tilde @ np.linalg.inv( cov ) * x_tilde ).sum()

    loglike = -0.5 * ( n * ( d * np.log( 2 * np.pi ) + cov_logdet ) + mhlnbs )
    
    return loglike



def cosine_similarity(
    x_normalzied,
    subspace_normalized,
):
    '''
    x:
        (1, D)
    subspace:
        (M, D) M orthornormal vectors
    '''
    x_normalzied_projection = x_normalzied @ subspace_normalized.T @ subspace_normalized

    x_normalzied_projection_normalized = x_normalzied_projection / (x_normalzied_projection**2).sum()**0.5

    sim = x_normalzied @ x_normalzied_projection_normalized.T
    sim = float(sim)

    return sim


def function_sample_scale_percentile(
    function_sample_list,
    q,
    ):
    function_sample_list_all = []
    for y in function_sample_list:
        function_sample_list_all += y.flatten().tolist()
    return np.percentile( np.abs(function_sample_list_all-np.mean(function_sample_list_all)), q=q)


def vectorize_matrix(m):
    return m.T.reshape(1,-1).T

def log_expect_gamma(a,b):
    '''
    expectation of ln(x) if x obeys gamma(a,b)
    '''
    return digamma(a)-np.log(b)
def gaussian_entropy(cov):
    return 0.5 * np.linalg.slogdet( 2 * np.pi * np.e * cov)[1]
def gaussian_entropy_covinv(covinv):
    '''
    gaussian_entropy using inverse of covariance matrix for numerical stability
    '''
    return - 0.5 * np.linalg.slogdet( ( 2 * np.pi * np.e )**(-1) * covinv)[1]
def gamma_entropy(a,b):
    return ( a - np.log(b) + loggamma(a) + (1-a)*digamma(a) )



def woodbury_inverse(
    a_inv,
    u,
    c_inv,
    v = None,
    return_parts = False,
):
    '''
    Woodbury matrix identity, ( a + u @ c @ v )**(-1) = a**(-1) - a**(-1) @ u @ ( c**(-1) + v @ a**(-1) @ u )**(-1) @ v @ a**(-1)
    ---
    return_parts:
        bool, whether to return intermediate results. If True, return (p1, p2, p3) so ( a + u @ c @ v )**(-1) = a**(-1) - p1 @ p2**(-1) @ p3
    a_inv:
        scalar or (m,m), inverse of a, if scalar, a_inv = 1/a, if (m,m), a_inv = np.linalg.inv(a)
    u:
        (m,n), m > n
    '''
    if v is None:
        v = u.T
    # return a_inv - a_inv @ u @ np.linalg.inv( c_inv + v @ a_inv @ u ) @ v @ a_inv
    if isinstance(a_inv, float):
        # return a_inv - u @ np.linalg.inv( c_inv @ a_inv**(-2) + v @ a_inv**(-1) * u ) @ v
        a = a_inv**(-1)
        if c_inv.ndim == 1:
            cvau = v @ u * a
            np.fill_diagonal( cvau, c_inv * a**2 + np.diag( cvau ) )
        else:
            cvau = c_inv * a**2 + v @ u * a
        if return_parts:
            
            return (u, cvau, v)

        else:
            cvaui = np.linalg.inv( cvau )
            r = - u @ cvaui @ v
            rd = np.diag( r )
            rd = a_inv + rd
            np.fill_diagonal( r, rd )

            return r
    else:
        aiu = a_inv @ u
        vai = v @ a_inv
        # if c_inv is a vector that is the diagonal of a matrix, use fill_diagonal for efficiency
        if c_inv.ndim == 1:
            cvau = vai @ u
            np.fill_diagonal( cvau, c_inv + np.diag( cvau ) )
        else:
            cvau = c_inv + vai @ u
        if return_parts:
            return (aiu, cvau, vai)
        else:
            cvaui = np.linalg.inv( cvau )
            acv = aiu @ cvaui @ vai
            r = a_inv - acv
            return r


def woodbury_logdet(
        a_e,
        u,
        w_inv,
        vt,
):
    '''
    logdet of ( diag( a_e ) - u @ inv(w_inv) @ vt ), m > n
    ---
    a_e:
        scalar, diagonal element of a
    u:
        (m,n)
    w_inv:
        (n,n)
    vt:
        (n,m)
    '''

    m,n = u.shape

    w = np.linalg.inv( w_inv )

    r0 = w @ (vt @ u) * (- a_e**(-1))
    np.fill_diagonal( r0, np.diag( r0 ) + 1 )

    r = np.linalg.slogdet( r0 )[1] + m * np.log( a_e )
    return  r




def svd_inverse(
        matrix, 
        cond_max=None, 
        jitter= 0.0, 
        hermitian = False,
        threshold_cond_num_warning = 1e10,
        logging_prefix = '',
        rcond_pinv = None,
        ):
    '''
    matrix:
        2d numpy array, square matrix
    threshold_cond_num_warning:
        if condition number is larger than this, a warning will be raised
    rcond_pinv:
        if not None, will return pseudo inverse in addition to inverse, rcond_pinv is the cutoff for small singular values, singular values smaller than rcond_pinv * largest_singular_value are considered zero
    ---
    cond_max:
        allowed maximum conditional number, for numerical stability.
    '''
    if jitter > 0:
        u,s,vh = np.linalg.svd(matrix + np.identity(n=matrix.shape[0], dtype= matrix.dtype) * jitter, hermitian= True)
    else:
        u,s,vh = np.linalg.svd(matrix, hermitian= True)

    if rcond_pinv is not None:
        cutoff = rcond_pinv * s.max()
        large_s = s > cutoff
        s_pinv = np.zeros_like(s)
        s_pinv[large_s] = s[large_s] ** (-1)
        m_pinv = u * s_pinv[None,:] @ u.T


    cond_num_l2 = s.max() / s.min()
    logging.debug(f'{logging_prefix}cond_num_l2 = {cond_num_l2}')
    if cond_num_l2 > threshold_cond_num_warning:
        logging.warning(f'{logging_prefix}cond_num_l2 = {cond_num_l2} is too large, svd_inverse may not be accurate.')

    if cond_max is not None:
        s += s.max() / cond_max
    m_inv = u / s[None,:] @ u.T

    if rcond_pinv is None:
        return m_inv
    else:
        return m_inv, m_pinv




class sinKernel():
    def __init__(self, length_scale) -> None:
        self.length_scale = length_scale
        pass

    def __call__(self, x1, x2) -> float:
        '''
        x1:
            (n_x1, )

        x2:
            (n_x2, )

        '''
        return np.sin( 2 * np.pi *( (x1[..., None] - x2[None, ...]) / self.length_scale) **2 )

class RBFKernel():
    def __init__(self, sigma, length_scale, multiplier= 1) -> None:
        self.sigma = sigma
        self.length_scale = length_scale
        self.multiplier = multiplier

    def __call__(self, x1, x2) -> float:
        '''
        x1:
            (n_x1, )

        x2:
            (n_x2, )

        '''
        return self.multiplier * np.exp(- (x1[..., None] - x2[None, ...]) ** 2 / ( 2 * self.length_scale**2 ))


class RBFKernel2():
    '''
    RBF kernel for data with multiple dimensions
    '''
    def __init__(self, sigma, length_scale, multiplier= 1) -> None:
        self.sigma = sigma
        self.length_scale = length_scale
        self.multiplier = multiplier

    def __call__(self, x1, x2) -> float:
        '''
        x1:
            (n_x1, n_dim)

        x2:
            (n_x2, n_dim)

        '''

        # offset.shape = (n_x1, n_x2, n_dim)
        offset = np.expand_dims(x1, axis=1) - np.expand_dims(x2, axis=0)
        dist = (offset**2).sum(axis=-1)

        return self.multiplier * np.exp(- dist / ( 2 * self.length_scale**2 ))
        


def lotlong2dist(x1, x2):
    '''
    calculate the distance on the sphere, geodesic distance using haversine formula
    ---
    x1:
        (n_x1, 2), latitude and longitude in degrees

    x2:
        (n_x2, 2), latitude and longitude in degrees
    ---
    return:
        (n_x1, n_x2), distance matrix, geodesic distance in km
    '''

    x1_lat, x1_lon = np.radians(x1[:,0]), np.radians(x1[:,1])
    x2_lat, x2_lon = np.radians(x2[:,0]), np.radians(x2[:,1])
    x1_lat = x1_lat[:, None]
    x1_lon = x1_lon[:, None]
    x2_lat = x2_lat[None, :]
    x2_lon = x2_lon[None, :]
    d_lat = ne.evaluate("x1_lat - x2_lat")
    d_lon = ne.evaluate("x1_lon - x2_lon")
    c1lt = np.cos(x1_lat)
    c2lt = np.cos(x2_lat)
    c1lt_c2lt = ne.evaluate("c1lt * c2lt")
    # hav_theta = np.sin(d_lat/2) ** 2 + c1lt_c2lt * np.sin(d_lon/2) ** 2
    hav_theta = ne.evaluate("sin(d_lat/2) ** 2 + c1lt_c2lt * sin(d_lon/2) ** 2")
    d = 2 * np.arcsin(np.sqrt(hav_theta))
    # radius of the Earth in km
    # dist = d * 6371
    # return dist
    d *= 6371
    return d

class KernelPresLatLonTime(Kernel):
    '''
    RBF kernel for data with 4 dimensions: pressure, latitude, longitude, and time
    '''
    def __init__(
        self,
        sigma=1.0,
        length_scale_geo=1.0,
        length_scale_pres=1.0,
        length_scale_time=1.0,
        length_scale_time_rbf=1.0,
        period_time=1.0,
        multiplier=1.0,
        sigma_bounds="fixed",
        length_scale_geo_bounds="fixed",
        length_scale_pres_bounds="fixed",
        length_scale_time_bounds="fixed",
        length_scale_time_rbf_bounds="fixed",
        period_time_bounds="fixed",
        multiplier_bounds="fixed",
    ):
        self.sigma = sigma
        self.length_scale_geo = length_scale_geo
        self.length_scale_pres = length_scale_pres
        self.length_scale_time = length_scale_time
        self.length_scale_time_rbf = length_scale_time_rbf
        self.period_time = period_time
        self.multiplier = multiplier

        self.sigma_bounds = sigma_bounds
        self.length_scale_geo_bounds = length_scale_geo_bounds
        self.length_scale_pres_bounds = length_scale_pres_bounds
        self.length_scale_time_bounds = length_scale_time_bounds
        self.length_scale_time_rbf_bounds = length_scale_time_rbf_bounds
        self.period_time_bounds = period_time_bounds
        self.multiplier_bounds = multiplier_bounds

    def __call__(
            self, 
            x1 = None,
            x2 = None,
            X = None, 
            Y = None,
            eval_gradient=False,
            batch_size = 500,
            ):
        '''
        X and Y are added for compatibility with sklearn.gaussian_process.kernels.Kernel. At least one of x1 and X must be specified.
        ---
        x1, X:
            (n_x1, 4), pressure, latitude, longitude, and time
        x2, Y:
            (n_x2, 4), pressure, latitude, longitude, and time
        batch_size:
            int, batch size of X1 to calculate kernel matrix to avoid memory error
        '''

        assert (x1 is not None) or (X is not None), 'either x1 or X must be specified'
        if x1 is None:
            x1 = X
        if x2 is None:
            x2 = x1 if Y is None else Y

        n_x1 = x1.shape[0]
        n_x2 = x2.shape[0]

        # Prepare final kernel matrix
        K_full = np.zeros((n_x1, n_x2), dtype=np.float64)

        pi = np.pi
        lsp_rcp = 1 / self.length_scale_pres
        lsg_rcp = 1 / self.length_scale_geo
        lst_rcp = 1 / self.length_scale_time
        # lst_rcp_rbf = 1 / self.length_scale_time_rbf
        pt_rcp = 1 / self.period_time
        mult = self.multiplier

        for start in range(0, n_x1, batch_size):
            end = min(start + batch_size, n_x1)
            x1_chunk = x1[start:end]
            if n_x1 > batch_size:
                logging.info(f'calculating kernel matrix for {start}:{end} of {n_x1}')

            dist_geo = lotlong2dist(x1_chunk[:, 1:3], x2[:, 1:3])

            x1_pres = x1_chunk[:, 0][:, None]
            x2_pres = x2[:, 0][None, :]
            x1_time = x1_chunk[:, 3][:, None]
            x2_time = x2[:, 3][None, :]

            dist_pres = ne.evaluate('abs(x1_pres - x2_pres)')
            dist_time = ne.evaluate('abs(x1_time - x2_time)')

            K_view = K_full[start:end, :]
            K_view[:] = np.exp(-0.5 * (dist_pres * lsp_rcp) ** 2)
            K_view *= np.exp(-0.5 * (dist_geo * lsg_rcp) ** 2)
            sin_part = np.sin(pi * dist_time * pt_rcp) * lst_rcp
            K_view *= np.exp(-2 * sin_part ** 2)
            # K_view *= np.exp(-0.5 * (dist_time * lst_rcp_rbf) ** 2)
            K_view *= mult

        if eval_gradient:
            # Since all hyperparameters are fixed, return empty gradient
            return K_full, np.empty((n_x1, n_x2, 0))
        else:
            return K_full

    def diag(self, X):
        return np.full(X.shape[0], self.multiplier)

    def is_stationary(self):
        return True

    def get_params(self, deep=True):
        return {
            "sigma": self.sigma,
            "length_scale_geo": self.length_scale_geo,
            "length_scale_pres": self.length_scale_pres,
            "length_scale_time": self.length_scale_time,
            "length_scale_time_rbf": self.length_scale_time_rbf,
            "period_time": self.period_time,
            "multiplier": self.multiplier
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    @property
    def hyperparameter_sigma(self):
        return Hyperparameter(
            "sigma", "numeric", self.sigma_bounds, 1, fixed=self.sigma_bounds == "fixed"
        )

    @property
    def hyperparameter_length_scale_geo(self):
        return Hyperparameter(
            "length_scale_geo",
            "numeric",
            self.length_scale_geo_bounds,
            1,
            fixed=self.length_scale_geo_bounds == "fixed",
        )

    @property
    def hyperparameter_length_scale_pres(self):
        return Hyperparameter(
            "length_scale_pres",
            "numeric",
            self.length_scale_pres_bounds,
            1,
            fixed=self.length_scale_pres_bounds == "fixed",
        )

    @property
    def hyperparameter_length_scale_time(self):
        return Hyperparameter(
            "length_scale_time",
            "numeric",
            self.length_scale_time_bounds,
            1,
            fixed=self.length_scale_time_bounds == "fixed",
        )
    
    @property
    def hyperparameter_length_scale_time_rbf(self):
        return Hyperparameter(
            "length_scale_time_rbf",
            "numeric",
            self.length_scale_time_rbf_bounds,
            1,
            fixed=self.length_scale_time_rbf_bounds == "fixed",
        )

    @property
    def hyperparameter_period_time(self):
        return Hyperparameter(
            "period_time",
            "numeric",
            self.period_time_bounds,
            1,
            fixed=self.period_time_bounds == "fixed",
        )

    @property
    def hyperparameter_multiplier(self):
        return Hyperparameter(
            "multiplier",
            "numeric",
            self.multiplier_bounds,
            1,
            fixed=self.multiplier_bounds == "fixed",
        )
    

class PresLatLonTimeFunction:
    def __init__(self, 
                 sigma, 
                 length_scale_geo, 
                 length_scale_pres, 
                 length_scale_time, 
                 period_time,
                 multiplier,
                 center,
                ):
        '''
        sigma: float
        length_scale_geo: float
        length_scale_pres: float
        length_scale_time: float
        period_time: float
        multiplier: float
        center: np.array, shape=(1, 4)
        '''
        self.sigma = sigma
        self.length_scale_geo = length_scale_geo
        self.length_scale_pres = length_scale_pres
        self.length_scale_time = length_scale_time
        self.period_time = period_time
        self.multiplier = multiplier
        self.center = center

    def __call__(self, x):
        '''
        x: np.array, shape=(n, d_index_set)
        '''
        fk = KernelPresLatLonTime(sigma=self.sigma, length_scale_geo=self.length_scale_geo, length_scale_pres=self.length_scale_pres, length_scale_time=self.length_scale_time, period_time=self.period_time, multiplier=self.multiplier)
        return fk(x1=x, x2=self.center)
                 

class PresLatLonTimeFunctionMixture:
    def __init__(self, lat_lon_pres_time_function_list):
        '''
        lat_lon_pres_time_function_list: list of PresLatLonTimeFunction objects
        '''
        self.lat_lon_pres_time_function_list = lat_lon_pres_time_function_list

    def dist2subspace(self, lat_lon_pres_time_function):
        '''
        distance between a PresLatLonTimeFunction object and the subspace spanned by the current list of PresLatLonTimeFunction objects
        ---
        lat_lon_pres_time_function: PresLatLonTimeFunction object
        '''
        center_ary = np.concatenate([l.center for l in self.lat_lon_pres_time_function_list], axis=0)
        dist_pres = np.abs(lat_lon_pres_time_function.center[:,0] - center_ary[:,0]).min()
        dist_time = np.abs(lat_lon_pres_time_function.center[:,3] - center_ary[:,3]).min()
        dist_geo = lotlong2dist(lat_lon_pres_time_function.center[:,:2], center_ary[:,:2]).min()
        dist = np.array([ dist_pres, dist_geo, dist_time ])

        return dist
    
    def check_if_similar(self, lat_lon_pres_time_function, threshold):
        '''
        check if a PresLatLonTimeFunction object is similar to the current list of PresLatLonTimeFunction objects
        ---
        lat_lon_pres_time_function: PresLatLonTimeFunction object
        threshold: (3, ) array, threshold for distance in pressure, geodesic distance, and time
        '''
        dist = self.dist2subspace(lat_lon_pres_time_function)
        center_ary = np.concatenate([l.center for l in self.lat_lon_pres_time_function_list], axis=0)
        dist_pres = np.abs(lat_lon_pres_time_function.center[:,0] - center_ary[:,0])
        dist_time = np.abs(lat_lon_pres_time_function.center[:,3] - center_ary[:,3])
        dist_geo = lotlong2dist(lat_lon_pres_time_function.center[:,:2], center_ary[:,:2])[0]

        dist= np.array([dist_pres, dist_geo, dist_time]).T

        is_similar = (dist < threshold[None,:]).all(axis=1).any()
        
        return is_similar


class PolyKernel():
    def __init__(self, degree = 1, sigma = 0, length_scale = 1) -> None:
        self.degree = degree
        self.sigma = sigma
        self.length_scale = length_scale

    def __call__(self, x1, x2) -> float:
        '''
        x1:
            (n_x1, )

        x2:
            (n_x2, )

        '''
        x1 = ( x1.copy() ) / self.length_scale
        x2 = ( x2.copy() ) / self.length_scale

        return np.exp( ( x1[..., None] * x2[None, ...] ) ** self.degree ) + self.sigma ** 2


class CpdKernel():
    def __init__(self, ) -> None:
        pass

    def __call__(self, x1, x2) -> float:
        '''
        x1:
            (n_x1, )

        x2:
            (n_x2, )

        '''
        return - np.abs( x1[..., None] - x2[None, ...] ) ** 2



class LogKernel():
    def __init__(self, b, length_scale = 1) -> None:
        self.b = b
        self.length_scale = length_scale

    def __call__(self, x1, x2) -> float:
        '''
        x1:
            (n_x1, )

        x2:
            (n_x2, )

        '''
        return - np.log( 1 + np.abs( ( x1[..., None] - x2[None, ...] ) / self.length_scale ) ** self.b )

class normalKernel():
    '''
    it provides normalized basis functions
    '''
    def __init__(
        self,
        kernel,
        x1_range,
        n_sample=100,
        )->None:
        self.kernel= kernel
        self.x1_range= x1_range
        self.n_sample= n_sample

    def __call__(self, x1, x2) -> float:
        '''
        x1:
            (n_x1, )

        x2:
            (n_x2, )

        '''
        r = self.kernel(x1, x2)

        l_grid = self.x1_range[1] - self.x1_range[0]
        x_grid = np.linspace(
            start= self.x1_range[0] + l_grid/self.n_sample/2, 
            stop= self.x1_range[1] - l_grid/self.n_sample/2, 
            num= self.n_sample)

        r_grid = self.kernel(x1, x_grid)

        integral_p2 = np.sum(r_grid**2 * (l_grid/self.n_sample), axis=1)

        return r / (integral_p2**0.5)[:,None]


class UniLinSplKernel():
    def __init__(self, length_scale=1, amplifier= 1, shift= 0) -> None:
        self.length_scale = length_scale
        self.amplifier = amplifier
        self.shift = shift
        pass

    def __call__(self, x1, x2) -> float:
        '''
        x1:
            (n_x1, )

        x2:
            (n_x2, )

        '''
        x1 = ( x1.copy() - self.shift ) / self.length_scale
        x2 = ( x2.copy() - self.shift ) / self.length_scale
        
        x1x2_min = np.min(  
            np.concatenate(
                [ 
                    ( x1[..., None] + 0 * x2[None, ...] ) [..., None], 
                    ( 0 * x1[..., None] + x2[None, ...] ) [..., None]
                ],
                axis= -1,
            ),
            axis= -1, 
        )
        r= 1 + x1[..., None] * x2[None, ...] * ( 1 + x1x2_min ) - ( x1[..., None] + x2[None, ...] ) / 2 * x1x2_min**2 + x1x2_min**3 / 3
        r = r * self.amplifier
        return r



def cv_kernel(
    kernel_list,
    x_up_list,
    y_up_list,
    kfold = 5,
    repeat=1,
    ):
    '''
    use k fold cross validation to select the optimal kernel for gaussian process regression 
    '''

    logging.info(f'starting cv_kernel')

    kernel_sqerror_list = [0] * len(kernel_list)
    for ik, kernel in enumerate(kernel_list):
        for i in range(repeat):
            ix_list = [
                list(range(len(x.T)))
                for x in x_up_list
            ]
            for ixi, ix in enumerate(ix_list):
                np.random.shuffle(ix_list[ixi])
            for j in range(kfold):
                for p, xp in enumerate(x_up_list):

                    if len(xp.T) >= kfold:
                    
                        kold_size = int(np.round(len(xp.T)/kfold))
                        
                        if j < kfold-1:
                            ix_list_p_test = ix_list[p][ j*kold_size : (j+1)*kold_size ]
                        else:
                            ix_list_p_test = ix_list[p][ j*kold_size : ]
                        ix_list_p_train = [ix for ix in ix_list[p] if ix not in ix_list_p_test]

                        if len(ix_list_p_train)>0 and len(ix_list_p_test)>0:

                            gp = GaussianProcessRegressor(kernel= kernel)
                            gp.fit(X= xp[:,ix_list_p_train].T, y= y_up_list[p][:,ix_list_p_train].T )

                            pred_test = gp.predict(X= xp[:,ix_list_p_test].T)

                            sqerror = ((pred_test - y_up_list[p][:,ix_list_p_test].T.flatten() )**2).sum()

                            kernel_sqerror_list[ik] += sqerror
        
        logging.info(f'squred error of {kernel}: {kernel_sqerror_list[ik] :.8e}')

    kernel_sqerror_list_rmse = (np.array(kernel_sqerror_list) / np.sum([len(x.T) for x in x_up_list]) / repeat)**0.5
    ik_min = np.argmin(kernel_sqerror_list_rmse)

    logging.info(f'np.argmin(kernel_sqerror_list_rmse)={ik_min}, kernel_list[ik_min]={kernel_list[ik_min]}, \nkernel_sqerror_list_rmse={pprint.pformat(sorted(list(zip(kernel_sqerror_list_rmse, kernel_list))))}')


    return kernel_list[ik_min]


def rbf_integral(
        length_scale,
        a,
    ):
    '''
    integral of rbf kernel from -a to a, i.e. \int_{-a}^a exp(-0.5(x/l)^2) dx
    '''

    return ( ( np.pi/2 )**0.5 * length_scale * math.erf( (a) / (2**0.5*length_scale) ) * 2 )


def cv_kernel_kmeans(
    length_scale_list,
    x_up_list,
    y_up_list,
    normalize_rbf,
    kfold = 5,
    repeat=1,
    n_clusters= 3,
    path_fig=None,
    multipier_constant_value= 3,
    x_domain_length= None,
    ):
    '''
    use k fold cross validation to select the optimal kernel for gaussian process regression 
    ---
    x_up_list:
        list of x_up (1, n_x_up_i, d_index_set) or (1, n_x_up_i) 
    y_up_list:
        list of y_up (1, n_y_up_i)
    '''

    logging.info(f'starting cv_kernel')

    x_up_list2 = copy.deepcopy(x_up_list)
    if x_up_list[0].ndim == 2:
        x_up_list2 = [x[...,None] for x in x_up_list2]

    x_up_lengthscale_error = np.zeros(shape=(len(x_up_list2), len(length_scale_list)))
    for ik, ls in enumerate(length_scale_list):

        logging.info(f'ik={ik}, ls={ls}')

        kernel = RBF(length_scale= ls, length_scale_bounds="fixed") * ConstantKernel(constant_value=multipier_constant_value/ rbf_integral(length_scale=ls,a=x_domain_length/2) if normalize_rbf else multipier_constant_value, constant_value_bounds="fixed" ) + WhiteKernel()

        for i in range(repeat):
            ix_list = [
                list(range( x.shape[1] ))
                for x in x_up_list2
            ]
            for ixi, ix in enumerate(ix_list):
                np.random.shuffle(ix_list[ixi])
            for j in range(kfold):
                for p, xp in enumerate(x_up_list2):

                    if xp.shape[1] >= kfold:
                    
                        kold_size = int(np.round(xp.shape[1]/kfold))
                        
                        if j < kfold-1:
                            ix_list_p_test = ix_list[p][ j*kold_size : (j+1)*kold_size ]
                        else:
                            ix_list_p_test = ix_list[p][ j*kold_size : ]
                        ix_list_p_train = [ix for ix in ix_list[p] if ix not in ix_list_p_test]

                        if len(ix_list_p_train)>0 and len(ix_list_p_test)>0:

                            gp = GaussianProcessRegressor(
                                kernel= kernel,
                                # NOTE this is necessary for non standard normal data
                                normalize_y= True,
                                )
                            gp.fit(X= xp[0,ix_list_p_train], y= y_up_list[p][:,ix_list_p_train].T )

                            pred_test = gp.predict(X= xp[0,ix_list_p_test])

                            sqerror = ((pred_test - y_up_list[p][:,ix_list_p_test].T.flatten() )**2).sum()

                            x_up_lengthscale_error[p][ik] += sqerror

    k2mse_per_point = sorted(
        list(zip( length_scale_list, x_up_lengthscale_error.mean(axis=0) / (repeat*kfold*np.sum([p.shape[1] for p in x_up_list2])) )), 
        key= lambda e:e[1] )
    logging.info(f'k2mse_per_point = sorted( list(zip( length_scale_list, x_up_lengthscale_error.mean(axis=0) / (repeat*kfold*np.sum([p.shape[1] for p in x_up_list])) )), key= lambda e:e[1] ) = {pprint.pformat( k2mse_per_point, indent= 4 )}')


    ls_opt_list = []
    for p,ls_err in enumerate(x_up_lengthscale_error):
        ls_opt_list.append( length_scale_list[np.argmin(ls_err)] )


    km = KMeans(
        n_clusters= n_clusters,
        init= 'k-means++',
    )

    km.fit(X= np.log10(np.array(ls_opt_list))[:,None])

    log10ls_cddt = np.sort(km.cluster_centers_.flatten())

    if path_fig is not None:
        fig,ax = plt.subplots(1,1)
        ax.hist( np.log10(np.array(ls_opt_list)), label= 'length_scale_histogram')
        ax.set_xlabel(f'log10(length_scale)')
        ax.set_ylabel(f'number of functions')
        ax.set_title(f'Clustering of length scales in the histogram')
        ax.legend()
        for i,ls in enumerate(log10ls_cddt):
            ax.axvline(x= ls, label= f'center_{i}', c = 'g', ls= '-')
        fig.savefig(path_fig)


    return 10**log10ls_cddt