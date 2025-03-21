'''
self adaptive funtional principal component analysis, bayesian scalable functional data analysis with a faster algorithm

this is based on the conditional independence of the columns of the weight matrix
'''

import logging
import numpy as np
import matplotlib.pylab as plt
import pprint
import os
import copy
from sklearn.cluster import KMeans, MiniBatchKMeans
from collections import OrderedDict
import os
import dill
import time
import resource
import numexpr as ne
import zlib

from mpl_toolkits.axes_grid1 import make_axes_locatable

from util import EPS, svd_inverse, woodbury_inverse, vectorize_matrix, log_expect_gamma, gaussian_entropy, gamma_entropy, woodbury_logdet
from util import loglike_normal, loglike_gamma, KernelPresLatLonTime, PresLatLonTimeFunction, PresLatLonTimeFunctionMixture
from rbf import RadicalBasisFunctionMixture, RadialBasisFunction, RBFKernel2
from nrvm import nRVM, KernelPresLatLonTime_threshold
logging.critical(f'KernelPresLatLonTime_threshold={KernelPresLatLonTime_threshold}')


def load_bsfda(path_bsfda):
    '''
    load the BSFDA object from a file, decompress the compressed attribute
    '''
    with open(path_bsfda, 'rb') as f:
        bsfda = dill.load(f)
    if hasattr(bsfda, 'cov_vecwup_active_compresed'):
        cov_vecwup_active = np.frombuffer(zlib.decompress(bsfda.cov_vecwup_active_compresed), dtype= bsfda.mu_vecwup_active.dtype).reshape(bsfda.mu_vecwup_active.shape[1], bsfda.mu_vecwup_active.shape[1])
        bsfda.cov_vecwup_active = cov_vecwup_active
        delattr(bsfda, 'cov_vecwup_active_compresed')
    return bsfda

def get_posterior_probability(
    z_up_list,
    w_up,
    z_up_bar,

    alpha,
    beta,
    eta,
    var_white,

    y_up_list,
    phi_up_list,

    a0,
    b0,
    ):
    # TODO, HACK, this function currently does not calculate the posterior probability
    return 1

    '''
    pr[ z, w, zbar, alpha, beta, eta, var_white | y, a0, b0 ]
        = pr[ y | z, w, zbar, alpha, beta, eta, var_white, a0, b0 ] * pr[ z, w, zbar, alpha, beta, eta, var_white | a0, b0 ]
        = pr[ y | z, w, zbar, var_white ] * pr[ z ] * pr[ w | alpha, beta ] * pr[ zbar | eta, beta ] * pr[ alpha, beta, eta, var_white | a0, b0 ]
    ---
    z_up_list:
        list of z_up, sdandard multivariate normal
    w_up:
        weight matrix
    z_up_bar:
        mean of z_up
    alpha:
        precision of rows of w_up
    beta:
        precision of columns of w_up
    eta:
        precision of z_up_bar
    var_white:
        variance of white noise
    y_up_list:
        list of y_up, observed data
    phi_up_list:
        list of phi_up, basis functions
    a0:
        shape parameter of gamma prior of alpha, beta, eta, var_white
    b0:
        rate parameter of gamma prior of alpha, beta, eta, var_white
    '''
    n_basis = alpha.shape[0]

    p1 = np.sum([
        loglike_normal(
            x = yi,
            mu = ( z_up_list[i] @ w_up + z_up_bar )@ phi_up_list[i],
            var_log = np.log(var_white),
        ) + loglike_normal(
            x = z_up_list[i],
            mu = None,
            var_log = 0,
        )
        for i,yi in enumerate( y_up_list )
    ])

    p2 = np.array([
        [
            loglike_normal(
                x = w_up[j,k][None,None],
                mu = None,
                var_log = - ( np.log(alpha[j]) + np.log(beta[k]) ),
            )
            for k in range(n_basis)
        ] for j in range(n_basis)
    ]).sum()

    p3 = np.sum([
        loglike_normal(
            x = z_up_bar[0,k][None,None],
            mu = None,
            var_log = - ( np.log(eta) + np.log(beta[k]) ),
        )
        for k in range(n_basis)
    ])

    p4a = np.sum([
        loglike_gamma(
            x = alpha[j],
            a = a0,
            b = b0,
        )
        for j in range(n_basis)
    ])

    p4b = np.sum([
        loglike_gamma(
            x = beta[k],
            a = a0,
            b = b0,
        )
        for k in range(n_basis)
    ])

    p4az = loglike_gamma(
        x = eta,
        a = a0,
        b = b0,
    )

    p4w = loglike_gamma(
        x = var_white**(-1),
        a = a0,
        b = b0,
    )

    return (p1 + p2 + p3) + p4a + p4b + p4az + p4w



def get_lower_bound_active(
    mu_zi_list_active,
    cov_zi_list_active,

    mu_vecwup_active,
    cov_vecwup_active,

    mu_zbar_active,
    cov_zbar_active,

    a_alpha_active,
    b_alpha_active,

    a_beta_active,
    b_beta_active,

    a_eta,
    b_eta,

    a_white,
    b_white,

    y_up_list,

    yyt_list_sum,
    ypt_list_active,
    ppt_list_active,
    
    a0=0,
    b0=0,
    ):
    '''
    get lower bound of the posterior for the active part
    ---
    k_active_list:
        select all if None
    cov_vecwup_active_inv:
        inverse of cov_vecwup_active, for stability in calculation of the gasussian entropy when cov_vecwup_active is ill-conditioned
    '''
    # TODO, HACK, this function currently does not calculate the lower bound
    return 0

    n_basis_active = mu_zi_list_active[0].shape[1]
    # NOTE this function is now only valid for the active part

    alpha_active = a_alpha_active / b_alpha_active
    beta_active = a_beta_active / b_beta_active
    eta = a_eta / b_eta


    ''' variables below are all active '''

    sigma_n2 = a_white / b_white
    alpha_active = a_alpha_active/b_alpha_active
    beta_active = a_beta_active/b_beta_active
    w_up_active = mu_vecwup_active.reshape((n_basis_active, n_basis_active)).T
    ln2pi = np.log(2*np.pi)
    
    ppt_list_active_sum = (np.sum(ppt_list_active, axis=0))

    d_up_active = np.zeros(shape=(n_basis_active*n_basis_active, n_basis_active*n_basis_active))
    for j in range(n_basis_active):
        for k in range(n_basis_active):
            d_up_active[ (j+k*n_basis_active, ), : ] = vectorize_matrix(( w_up_active[ (k,), : ].T @ w_up_active[ (j,), : ] + cov_vecwup_active[ j::n_basis_active, k::n_basis_active ] ).T).T
    b_white_new = 0.5 * yyt_list_sum - w_up_active.T.reshape(1,-1) @ np.sum([
        mu_zi_list_active[i].T @ ypt_list_active[i]
        for i,yi in enumerate(y_up_list)
    ], axis=0).T.reshape(1,-1).T + 0.5 * vectorize_matrix( d_up_active.T ).T @ vectorize_matrix(np.sum([
        vectorize_matrix( ppt_list_active[i] ) @ vectorize_matrix( cov_zi_list_active[i] + mu_zi_list_active[i].T @ mu_zi_list_active[i] ).T
        for i,yi in enumerate(y_up_list)
    ], axis=0)) - np.sum([
        (ypt_list_active[i] - mu_zi_list_active[i] @ w_up_active @ ppt_list_active[i] ) @ mu_zbar_active.T
        for i,yi in enumerate(y_up_list)
    ]) + 0.5 * ( cov_zbar_active + mu_zbar_active.T @ mu_zbar_active ).T.reshape(1,-1) @ ppt_list_active_sum.T.reshape(1,-1).T
    b_white_new = float(b_white_new)

    lb1 = - sigma_n2 * b_white_new
    log_expect_gamma_white = log_expect_gamma(a=a_white,b=b_white)
    lb2a= np.sum([
                yi.shape[1] * ( ln2pi - log_expect_gamma_white ) + n_basis_active * ln2pi + np.sum( mu_zi_list_active[i][0]**2 + np.diag(cov_zi_list_active[i]) ) 
                for i,yi in enumerate(y_up_list)
        ])
    lb2b= np.sum([
            np.sum([
                ln2pi - log_expect_gamma(a=a_alpha_active[j],b=b_alpha_active[j]) - log_expect_gamma(a=a_beta_active[k],b=b_beta_active[k]) + alpha_active[j]*beta_active[k]*(w_up_active[j,k]**2 + cov_vecwup_active[k*n_basis_active+j, k*n_basis_active+j])
                for k in range(n_basis_active)
            ])
            for j in range(n_basis_active)
        ])

    lb2c = ( ln2pi * n_basis_active - np.sum([
        log_expect_gamma(a=a_beta_active[k],b=b_beta_active[k]) + log_expect_gamma(a=a_eta,b=b_eta)
        for k in range(n_basis_active)
    ]) +  ( ( np.diag( cov_zbar_active ) + mu_zbar_active[0]**2 ) * ( eta * beta_active) ).sum() )
    jk = np.zeros(shape=(3, n_basis_active, n_basis_active))
    for j in range(n_basis_active):
        for k in range(n_basis_active):
            jk[0,j,k] = ln2pi - log_expect_gamma(a=a_alpha_active[j],b=b_alpha_active[j]) - log_expect_gamma(a=a_beta_active[k],b=b_beta_active[k]) + alpha_active[j]*beta_active[k]*(w_up_active[j,k]**2 + cov_vecwup_active[k*n_basis_active+j, k*n_basis_active+j])
            jk[1,j,k] = log_expect_gamma(a=a_alpha_active[j],b=b_alpha_active[j])
            jk[2,j,k] = log_expect_gamma(a=a_beta_active[k],b=b_beta_active[k])
    lb3 = np.sum([
            gaussian_entropy(cov=cov_zi_list_active[i])
            for i,yi in enumerate(y_up_list)
        ])

    lb4 = gaussian_entropy(cov=cov_zbar_active)
    lb5 = gaussian_entropy(cov=cov_vecwup_active)

    lb6 = np.sum([
                        gamma_entropy(a=a_alpha_active[j], b=b_alpha_active[j])
                        for j in range(n_basis_active)
                    ])
    lb6_eta = gamma_entropy(a=a_eta, b=b_eta)
    lb7 = np.sum([
                            gamma_entropy(a=a_beta_active[k], b=b_beta_active[k])
                            for k in range(n_basis_active)
                        ])
    lb8 = gamma_entropy(a=a_white, b=b_white)


    lb9j = np.sum([
        (a0-1) * log_expect_gamma(a=a_alpha_active[j],b=b_alpha_active[j]) - b0 * alpha_active[j]
        for j in range(n_basis_active)
    ])
    lb9k = np.sum([
        (a0-1) * log_expect_gamma(a=a_beta_active[k],b=b_beta_active[k]) - b0 * beta_active[k]
        for k in range(n_basis_active)
    ]) 
    lb9z = (a0-1) * log_expect_gamma(a=a_eta,b=b_eta) - b0 * ( a_eta / b_eta )
    lb9s = (a0-1) * log_expect_gamma(a=a_white,b=b_white) - b0 * sigma_n2
    
    
    logging.debug(f'jk0={jk[0]}')
    logging.debug(f'jk1={jk[1]}')
    logging.debug(f'jk2={jk[2]}')
    logging.debug(f'a_alpha_active={a_alpha_active}')
    logging.debug(f'b_alpha_active={b_alpha_active}')
    logging.debug(f'a_beta_active={a_beta_active}')
    logging.debug(f'b_beta_active={b_beta_active}')
    logging.debug(f'lb1={lb1}')
    logging.debug(f'lb2a={float(lb2a)}')
    logging.debug(f'lb2b={float(lb2b)}')
    logging.debug(f'lb2c={float(lb2c)}')
    logging.debug(f'lb3={float(lb3)}')
    logging.debug(f'lb4={float(lb4)}')
    logging.debug(f'lb5={float(lb5)}')
    logging.debug(f'lb6={float(lb6)}')
    logging.debug(f'lb6_eta={float(lb6_eta)}')
    logging.debug(f'lb7={float(lb7)}')
    logging.debug(f'lb8={float(lb8)}')
    logging.debug(f'lb9j={float(lb9j)}')
    logging.debug(f'lb9k={float(lb9k)}')
    lower_bound = lb1 - 0.5 * ( lb2a + lb2b + lb2c ) \
        + ( lb3 + lb4 + lb5 + lb6 + lb6_eta + lb7 + lb8 ) \
            + ( lb9j + lb9k + lb9z + lb9s )
    lower_bound = float(lower_bound)
    
    return lower_bound


class BSFDA():
    '''
    Bayesian Scalable Functional Data Analysis
    '''
    def __init__(
        self,
        kernel_list,
        max_precision_active_alpha = EPS**(-0.5)/10,
        max_precision_active_beta = EPS**(-0.5)/10,
        precision_threshold_multiplier2min = EPS**(-0.5)/10,
        a0= np.finfo(float).eps,
        b0= np.finfo(float).eps,
        init_machine_precision = np.finfo(float).eps,
        cond_num_max = None,
        rcond_pinv = 1e-15,
        EPS=EPS,
        ) -> None:
        '''
        kernel_list:
            list of functions: ((n_x1, d_index), (n_x2d_index)) -> (n_x1, n_x2), for convenience it should be in the same scale of the measurements
        precision_threshold_multiplier2min:
            the multiplier of the minimum precision to threshold the precision
        max_precision_active_alpha, max_precision_active_beta:
            active precisions larger than this value will be thresholded to prevent numerical issues
        init_machine_precision:
            a small value for setting the parameters that have not been updated yet and used to be infinitesimal, e.g. covariance matrix.
        a0, b0:
            hyperparameters for the prior gamma distribution of the precision of the weights
        '''

        self.kernel_list = kernel_list
        self.precision_threshold_multiplier2min = np.float64(precision_threshold_multiplier2min) if type(precision_threshold_multiplier2min) is float else precision_threshold_multiplier2min
        self.lower_bound_history_list = []
        self.posterior_history_list = []
        # list of indices of lower bound history list where new basis is added
        self.n_lbhist_new_basis_list = []
        self.n_lbhist_remove_basis_list = []
        # list of indices of newly added basis functions
        self.k_list_newly_tried = []
        self.k_list_similar2active = []

        self.max_precision_active_alpha = np.float64(max_precision_active_alpha) if type(max_precision_active_alpha) is float else max_precision_active_alpha
        self.max_precision_active_beta = np.float64(max_precision_active_beta) if type(max_precision_active_beta) is float else max_precision_active_beta
        logging.info(f'self.max_precision_active_alpha={self.max_precision_active_alpha:.3e}')
        logging.info(f'self.max_precision_active_beta={self.max_precision_active_beta:.3e}')

        self.a0= a0 
        self.b0= b0
        
        self.EPS = EPS

        self.cond_num_max = cond_num_max
        self.rcond_pinv = rcond_pinv
        self.init_machine_precision= init_machine_precision

    def dump(self, path_dump, exclude_data=False):
        '''
        Use dill to dump the object, but exclude certain attributes.
        ---
        exclude_data:
            if True, exclude the data attributes to save space
        '''
        # List of attribute names to exclude from dumping
        attributes_to_exclude = [
            'x_up_list',
            'y_up_list',
        ]
        
        # Temporarily store the excluded attributes
        excluded_attributes = {attr: getattr(self, attr) for attr in attributes_to_exclude if hasattr(self, attr)}
        
        # Remove the attributes from the object
        for attr in attributes_to_exclude:
            if hasattr(self, attr):
                delattr(self, attr)

        # this attribute is too large and will be compressed
        cov_vecwup_active = self.cov_vecwup_active
        cov_vecwup_active_compresed = zlib.compress(cov_vecwup_active.tobytes())
        self.cov_vecwup_active_compresed = cov_vecwup_active_compresed
        delattr(self, 'cov_vecwup_active')
        
        # Dump the object
        with open(path_dump, 'wb') as f:
            dill.dump(self, f)
        
        # Restore the excluded attributes
        for attr, value in excluded_attributes.items():
            setattr(self, attr, value)
        self.cov_vecwup_active = cov_vecwup_active

    def get_idx_alpha_active_effective(self):
        '''
        get a list of indices of effective principal components. index is within the active components.
        '''
        alpha_active_expect = self.a_alpha_active /self.b_alpha_active
        threshold_alpha = min( (alpha_active_expect.min() * self.precision_threshold_multiplier2min), self.max_precision_active_alpha )
        idx_alpha_active_effective = np.where(
            np.logical_not(np.logical_or(
                alpha_active_expect > threshold_alpha,
                np.isclose(alpha_active_expect, threshold_alpha),
            ))
        )[0].tolist()

        return idx_alpha_active_effective

    def get_idx_beta_active_effective(self):
        '''
        get a list of indices of effective basis functions. index is within the active basis functions.
        '''
        beta_active_expect = self.a_beta_active /self.b_beta_active
        threshold_beta = min( (beta_active_expect.min() * self.precision_threshold_multiplier2min), self.max_precision_active_beta )
        idx_beta_active_effective = np.where(
            np.logical_not(np.logical_or(
                beta_active_expect > threshold_beta,
                np.isclose(beta_active_expect, threshold_beta),
            ))
        )[0].tolist()

        return idx_beta_active_effective




    def plot_results(
        self,

        x_up_d0_grid = None,
        d0_index_set_plot = 0,

        sigma_true=None,
        w_up_true=None,
        w_x_phi_up_list_true = None,
        mean_true = None,

        full_cov_mise_npts = -1,
        n_recent = 100,
        path_fig=None,

        n_max_point_per_func_display = 1000,
        n_max_uncertainty = 0,
        i_list = None,
        uncertainty_use_theta = True,
        ):
        '''
        evaluate and plot the results for a certain dimension d0 of the index set
        ---
        x_up_d0_grid:
            (n_grid), grid index set in a certain dimension for plotting the results and evaluating the mean integrated squared error
        d0_index_set_plot:
            index of the dimension for plotting the results
        sigma_true:
            true noise level
        w_up_true:
            (n_component_true, n_basis_true), true weights
        w_x_phi_up_list_true:
            (n_grid, d_index_set) -> (n_component_true, n_grid), true ( w_up @ phi )
            (n_component_true, n_grid), true ( w_up @ phi_up_grid )
        mean_true:
            true mean function, (n_grid, d_index_set) -> (n_grid)
        full_cov_mise_npts:
            number of points to calculate the mean integrated squared error of covariance in the full space. if -1, it will only calculate the mean integrated squared error in one dimension
        i_list:
            list of indices of the functions to plot, if None, first functions will be plotted
        n_max_uncertainty:
            number of functions to plot the uncertainty
        uncertainty_use_theta:
            whether to use theta in the uncertainty calculation, if False, use z and w
        ---
        return:
            mu_mise:
                mean integrated squred error approximated using the working grid
            cov_mise:
                mean integrated squred error approximated using the working grid
        '''


        # NOTE all the variables below are of finite beta
        alpha_active_expect = self.a_alpha_active /self.b_alpha_active
        threshold_alpha = min( (alpha_active_expect.min() * self.precision_threshold_multiplier2min), self.max_precision_active_alpha )
        idx_alpha_active_effective = np.where(
            np.logical_not(np.logical_or(
                alpha_active_expect > threshold_alpha,
                np.isclose(alpha_active_expect, threshold_alpha),
            ))
        )[0].tolist()
        if len(idx_alpha_active_effective)<1:
            idx_alpha_active_effective = [np.argmin(alpha_active_expect)]
            logging.warning(f'all alphas are too large and the smallest one will be used for plotting!')

        beta_active_expect = self.a_beta_active /self.b_beta_active
        threshold_beta = min( (beta_active_expect.min() * self.precision_threshold_multiplier2min), self.max_precision_active_beta )
        idx_beta_active_effective = np.where( beta_active_expect<threshold_beta )[0].tolist()
        if len(idx_beta_active_effective)<1:
            idx_beta_active_effective = [np.argmin(beta_active_expect)]
            logging.warning(f'all betas are too large and the smallest one will be used for plotting!')

        k_active_list = self.k_active_list
        k_effective_list = np.array(k_active_list)[ idx_beta_active_effective ].tolist()


        mu_zi_list_effective = [
            xi[:,idx_alpha_active_effective] 
            for i,xi in enumerate(self.mu_zi_list_active)
        ]
        cov_zi_list_effective = [
            ci[np.ix_(idx_alpha_active_effective, idx_alpha_active_effective)]
            for i,ci in enumerate(self.cov_zi_list_active)
        ]
        mu_theta_i_list_effective = [
            xi[:,idx_beta_active_effective] 
            for i,xi in enumerate(self.mu_theta_i_list_active)
        ]
        cov_theta_i_list_effective = [
            self.cov_theta_i_list_active[i][ np.ix_( idx_beta_active_effective, idx_beta_active_effective ) ]
            for i in range(len(self.cov_theta_i_list_active))
        ]

        mu_zbar_effective = self.mu_zbar_active[ :, idx_beta_active_effective ]
        cov_zbar_effective = self.cov_zbar_active[ np.ix_( idx_beta_active_effective, idx_beta_active_effective ) ]

        n_basis_active = len(k_active_list)
        n_index_basis = self.x_base_index.shape[0]

        sigma_true = np.nan if sigma_true is None else sigma_true


        d_index_set = self.x_base_index.shape[1]


        w_up_active = self.mu_vecwup_active.reshape((n_basis_active, n_basis_active)).T
        w_up_effective = w_up_active[ np.ix_( idx_alpha_active_effective, idx_beta_active_effective ) ]
        w_up_effective_cov = self.cov_vecwup_active.reshape((n_basis_active, n_basis_active, n_basis_active, n_basis_active))[ np.ix_( idx_beta_active_effective, idx_alpha_active_effective, idx_beta_active_effective, idx_alpha_active_effective ) ]
        
        sigma_expect = (self.a_white / self.b_white)**(-0.5)
        sigmak_expect = np.exp( np.log(self.a_sigmak_active) - np.log(self.b_sigmak_active) )**(-0.5)
        y_up_list = self.y_up_list
        x_up_list = self.x_up_list

        for i in range(len(y_up_list)):
            if y_up_list[i].shape[1] > n_max_point_per_func_display:
                logging.warning(f'number of points for function {i} is too large and will be truncated to {n_max_point_per_func_display}!')
                break

        x_up_list_mu_ni= np.array([(xi.mean(axis=1).flatten().tolist() + [xi.shape[1]]) for xi in x_up_list])
        x_up_list_mu = ( x_up_list_mu_ni[:, :-1] * x_up_list_mu_ni[:, [-1]] ).sum(axis=0) /  x_up_list_mu_ni[:, [-1]].sum()

        # NOTE only use the first dimension of index set for plotting, d0_index_set_plot, the other dimensions are zero
        x_up_list_d0 = [ x_up_list[i][..., d0_index_set_plot] for i in range(len(x_up_list)) ]

        # grid of basis functions that are effective
        phi_up_grid_effective = []
        if x_up_d0_grid is None:
            x_up_grid_max = np.array([ z.max() for z in x_up_list_d0]).max()
            x_up_grid_min = np.array([ z.min() for z in x_up_list_d0]).min()
            x_up_d0_grid = np.linspace(start=x_up_grid_min, stop=x_up_grid_max,)
        # (n_grid, d_index_set)
        x_up_grid = np.ones((x_up_d0_grid.shape[0], d_index_set))
        for d in range(d_index_set):
            if d == d0_index_set_plot:
                x_up_grid[:, d] = x_up_d0_grid
            else:
                index_center_d = self.x_base_index[:, d].mean()
                x_up_grid[:, d] *= index_center_d


        for k in k_effective_list:
            x_up_basis_k = self.x_base_index[ [k % n_index_basis] ]
            phi_up_grid_effective.append( self.kernel_list[ k // n_index_basis ] ( x_up_basis_k, x_up_grid ) )

        phi_up_grid_effective = np.concatenate( phi_up_grid_effective, axis=0 )

        # NOTE only use the first dimension of index set for plotting
        # grid of basis functions that are active
        phi_up_grid_active = []
        phi_up_grid_active_index_set_d0 = []

        for k in k_active_list:
            x_up_basis_k = self.x_base_index[ [k % n_index_basis] ]
            phi_up_grid_active.append( self.kernel_list[ k // n_index_basis ] ( x_up_basis_k, x_up_grid ) )
            phi_up_grid_active_index_set_d0.append( x_up_basis_k[0,d0_index_set_plot] )

        phi_up_grid_active = np.concatenate( phi_up_grid_active, axis=0 )



        fig, ax = plt.subplots(nrows=15, ncols=1, figsize=(16,100))

        # color cycle
        c_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

        i_list_ = [i for i in range( min(len(y_up_list), len(c_list)-1) )] if i_list is None else i_list
        mu_grid = (mu_zbar_effective @ phi_up_grid_effective).flatten()
        for i in i_list_:
            ax[0].scatter(
                x= x_up_list_d0[i].T[:n_max_point_per_func_display], 
                y=y_up_list[i].T[:n_max_point_per_func_display], 
                c= c_list[(i+1) % len(c_list)],
                s = ( plt.rcParams['lines.markersize'] ** 2 ) / 4,
                alpha= 0.9,
                )
            ax[0].plot(
                x_up_d0_grid,
                # ( ( ( mu_zi_list_active[i] ) @ w_up_effective + mu_zbar_effective) @ phi_up_grid_effective).flatten(),
                ( ( mu_theta_i_list_effective[i] + mu_zbar_effective) @ phi_up_grid_effective).flatten(),
                linestyle= '-.',
                c= c_list[(i+1) % len(c_list)],
            )
        if mean_true is not None:
            mean_true_d0_grid = mean_true(x_up_grid)
            ax[0].plot(
                x_up_d0_grid,
                mean_true_d0_grid,
                linestyle= '-',
                label= f'mean',
                alpha= 0.7,
                c= c_list[0 % len(c_list)],
            )
            # mean interated squred error
            mu_mise = ( ( mean_true_d0_grid - mu_grid )**2 ).mean()
        else:
            mu_mise = None
        ax[0].plot(
            x_up_d0_grid,
            mu_grid,
            linestyle= '--',
            label= f'mean_est',
            c= c_list[0 % len(c_list)],
        )
        # uncertainty
        ept_sigma2 = self.b_white / ( self.a_white - 1 )
        ept_zbtzb = mu_zbar_effective.T @ mu_zbar_effective + cov_zbar_effective
        ept_sigmak2 = self.b_sigmak_active / ( self.a_sigmak_active - 1 )
        # with theta
        for i in i_list_[ : n_max_uncertainty ]:
            if uncertainty_use_theta:
                c_ = cov_zbar_effective + cov_theta_i_list_effective[i] 
                est_var_ = ( phi_up_grid_effective.T @ c_ * phi_up_grid_effective.T ).sum(axis=1) + ept_sigma2
                est_std_ = est_var_**0.5

                yi_ept = ( ( mu_theta_i_list_effective[i] + mu_zbar_effective) @ phi_up_grid_effective )
            else:
                # with z and w
                d_ = ( mu_zi_list_effective[i].T @ mu_zi_list_effective[i] + cov_zi_list_effective[i])
                c_ = np.zeros( ( len(idx_beta_active_effective), len(idx_beta_active_effective) ) )
                for i1 in range( len(idx_beta_active_effective) ):
                    for i2 in range( len(idx_beta_active_effective) ):
                        c_[i1,i2] = np.trace( ( w_up_effective[:,[i1]] @ w_up_effective[:,[i2]].T + w_up_effective_cov[ i1, :, i2, : ] ) @ d_ )
                c_ += ept_zbtzb + mu_zbar_effective.T @ mu_zi_list_effective[i] @ w_up_effective * 2
                np.fill_diagonal(c_, c_.diagonal() + ept_sigmak2)
                # expecation of squared noise
                yi_ept = ( ( mu_zi_list_effective[i] ) @ w_up_effective + mu_zbar_effective) @ phi_up_grid_effective
                # for n in range(len(x_up_d0_grid)):
                #     est_var_[n] = phi_up_grid_effective[ :, [n] ].T @ c_ @ phi_up_grid_effective[ :, [n] ] + ept_sigma2 - yi_ept[0,n] ** 2
                est_var_ = ( phi_up_grid_effective.T @ c_ * phi_up_grid_effective.T ).sum(axis=1) + ept_sigma2 - (yi_ept**2).flatten()
                est_std_ = est_var_**0.5
                
            ax[0].fill_between(
                x_up_d0_grid,
                # (mu_zbar_effective @ phi_up_grid_effective).flatten() - est_var_**0.5,
                # (mu_zbar_effective @ phi_up_grid_effective).flatten() + est_var_**0.5,
                yi_ept.flatten() - est_std_ * 2,
                yi_ept.flatten() + est_std_ * 2,
                alpha= 0.3,
                color= c_list[(i+1) % len(c_list)],
                # label= f'uncertainty',
            )
        # mean uncertainty
        est_var_ = ( phi_up_grid_effective.T @ cov_zbar_effective * phi_up_grid_effective.T ).sum(axis=1)
        est_std_ = est_var_**0.5
        if n_max_uncertainty > 0:
            ax[0].fill_between(
                x_up_d0_grid,
                mu_grid - est_std_ * 2,
                mu_grid + est_std_ * 2,
                alpha= 0.3,
                color= c_list[0 % len(c_list)],
            )

        ax[0].set_xlabel(f'x')
        ax[0].set_ylabel(f'y')
        ax[0].set_title(f'first {min(len(y_up_list), len(c_list)-1)} observed and estimated functions (n={len(y_up_list)}, Ni ∈ {self.n_up_i_range}]). center at [{",".join([f"{e:.3e}" for e in x_up_list_mu.tolist()])}], d0_index_set_plot={d0_index_set_plot}')
        ax[0].legend()



        if w_up_true is not None:
            u,s,v=np.linalg.svd(w_up_true)
            for i, si in enumerate( s[ : 2*len(c_list) ] ):
                vi= v[i,:].T
                ax[1].plot(
                    vi * np.sign( vi[np.argmax(np.abs(vi))] ), 
                    label= f'eigf_{i}, std={s[i]:.3e}',
                    alpha= 0.7 if i < len(c_list) else 0.5,
                    )
            ax[1].legend()

        ax[1].set_title(f'eigenfunctions of W and sigma={sigma_true:.3e} (truth)')



        # covariance estimation
        e_wtw_up_effective = ( w_up_effective.T @ w_up_effective + w_up_effective_cov.sum(axis=(1,3)) )
        cov_hat = phi_up_grid_effective.T @ e_wtw_up_effective @ phi_up_grid_effective
        cax = (make_axes_locatable(ax[12])).append_axes('right', size='5%', pad=0.05)
        x,y = np.meshgrid(x_up_d0_grid,x_up_d0_grid)
        im = ax[12].contourf(
            x,
            y,
            cov_hat,
            cmap= 'YlOrRd'
        )
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax[12].set_title(f'covariance estimation')
        ax[12].set_box_aspect(1)


        if ( w_x_phi_up_list_true is not None ):
            
            w_x_phi_up_list_true_d0_grid = w_x_phi_up_list_true( x_up_grid )

            u,s_wt,v_wt=np.linalg.svd(w_x_phi_up_list_true_d0_grid)
            v_wt_norm = np.zeros_like(v_wt)
            for i, si in enumerate( s_wt[ : 2*len(c_list) ] ):
                # original basis functions
                vi= w_x_phi_up_list_true_d0_grid[i,:].T
                std = (vi**2).sum()**0.5
                vi = vi / std
                ax[3].plot(
                    x_up_d0_grid, 
                    vi * np.sign( vi[np.argmax(np.abs(vi))] ), 
                    c= c_list[i % len(c_list)],
                    label= f'normalized_WΦ_{i}, std={std:.3e}',
                    alpha= 1 if i < len(c_list) else 0.7,
                    )
                # orthogonal components
                vi= v_wt[i,:].T
                vi= vi * np.sign( vi[np.argmax(np.abs(vi))] )
                v_wt_norm[i,:]= vi.T
                ax[4].plot(
                    x_up_d0_grid, 
                    vi,
                    c= c_list[i % len(c_list)],
                    ls= '-',
                    alpha= 0.7 if i < len(c_list) else 0.5,
                    label= f'eigf{i}, std={s_wt[i]:.3e}'
                    )

            ax[3].legend()


            # covariance truth
            cov_true = w_x_phi_up_list_true_d0_grid.T @ w_x_phi_up_list_true_d0_grid
            cax = (make_axes_locatable(ax[13])).append_axes('right', size='5%', pad=0.05)
            x,y = np.meshgrid(x_up_d0_grid,x_up_d0_grid)
            im = ax[13].contourf(
                x,
                y,
                cov_true,
                cmap= 'YlOrRd'
            )
            fig.colorbar(im, cax=cax, orientation='vertical')
            ax[13].set_title(f'covariance truth')
            ax[13].set_box_aspect(1)


            # covariance error
            cov_err = np.abs( cov_true - cov_hat )
            cax = (make_axes_locatable(ax[14])).append_axes('right', size='5%', pad=0.05)
            x,y = np.meshgrid(x_up_d0_grid,x_up_d0_grid)
            im = ax[14].contourf(
                x,
                y,
                cov_err,
                cmap= 'YlOrRd'
            )
            fig.colorbar(im, cax=cax, orientation='vertical')
            ax[14].set_title(f'covariance error')
            ax[14].set_box_aspect(1)

            # mean interated squred error
            cov_mise = ( cov_err**2 ).mean()
        else:
            cov_mise = None


        if full_cov_mise_npts > 0:
            x_min = np.min([ xi[0].min(axis=0) for xi in x_up_list], axis=0)
            x_max = np.max([ xi[0].max(axis=0) for xi in x_up_list], axis=0)
            grid_ranges = [np.linspace(min_val, max_val, full_cov_mise_npts) for min_val, max_val in zip(x_min, x_max)]
            mesh = np.meshgrid(*grid_ranges, indexing='ij')
            # Combine grid points into a single array of shape (full_cov_mise_npts^d, d)
            x_up_grid_full = np.stack(mesh, axis=-1).reshape(-1, len(x_min))
    
            phi_up_full_grid_effective = []
            for k in k_effective_list:
                x_up_basis_k = self.x_base_index[ [k % n_index_basis] ]
                phi_up_full_grid_effective.append( self.kernel_list[ k // n_index_basis ] ( x_up_basis_k, x_up_grid_full ) )
            phi_up_full_grid_effective = np.concatenate( phi_up_full_grid_effective, axis=0 )

            # covariance error in the full space
            w_x_phi_up_list_true_full_grid = w_x_phi_up_list_true( x_up_grid_full )
            cov_full_true = w_x_phi_up_list_true_full_grid.T @ w_x_phi_up_list_true_full_grid
            cov_full_hat = phi_up_full_grid_effective.T @ e_wtw_up_effective @ phi_up_full_grid_effective
            cov_full_err = np.abs( cov_full_true - cov_full_hat )
            cov_full_mise = ( cov_full_err**2 ).mean()

            cov_mise = cov_full_mise


        u,s,v=np.linalg.svd( w_up_effective @ phi_up_grid_effective )
        for i, si in enumerate( s[ : 2*len(c_list) ] ):
            # if alpha_active[i] < np.inf:
            vi= v[i,:].T
            if ( w_x_phi_up_list_true is not None ) and ( i <= (len(s_wt)-1) ):
                # choose the sign that minimizes the squared errors between the estimated and the true eigen function
                vi1= vi * (-1)
                if ((vi- v_wt_norm[i,:].T)**2).sum() > ((vi1- v_wt_norm[i,:].T)**2).sum():
                    vi = vi1
            else:
                # choose the sign such that the vertex is positive
                vi= vi * np.sign( vi[np.argmax(np.abs(vi))] )
            ax[4].plot(
                x_up_d0_grid, 
                vi,
                c= c_list[i % len(c_list)],
                ls= '--',
                alpha= 1 if i < len(c_list) else 0.7,
                label= f'eigf{i}_est, std={s[i]:.3e}')


        ax[3].set_xlabel(f'x')
        ax[3].set_ylabel(f'y')
        ax[3].set_title(f'original WΦ, sigma={sigma_true:.3e}')

        ax[4].set_xlabel(f'x')
        ax[4].set_ylabel(f'y')
        ax[4].set_title(f'eigenfunctions of WΦ, sigma={sigma_true:.3e}, sigma_est={sigma_expect:.3e}, sigmak_expect[0]={sigmak_expect[0]:.3e}')
        ax[4].legend()


        u,s,v=np.linalg.svd( w_up_effective )
        for i, si in enumerate( s[ : 2*len(c_list) ] ):
            # if alpha_active[i] < np.inf:
            vi= v[i,:].T
            ax[2].plot(
                vi * np.sign( vi[np.argmax(np.abs(vi))] ), label= f'eigf_est{i}, std={s[i]:.3e}', ls= '--',
                alpha= 1 if i < len(c_list) else 0.7,
                )

        ax[2].set_title(f'eigenfunctions of W, sigma={sigma_expect:.3e} (estimation)')
        ax[2].legend()


        for i_sort, i in enumerate(np.argsort( beta_active_expect )[ : 2*len(c_list) ]):
            p = phi_up_grid_active[i,:]
            ax[5].plot(
                x_up_d0_grid, 
                p, 
                label=f'{i_sort},x[{i_sort}]={phi_up_grid_active_index_set_d0[i]:.3e},log10(beta[{i_sort}])={np.log10(beta_active_expect[i]):.3f}', 
                c= c_list[i_sort % len(c_list)],
                alpha= 1 if (i_sort < len(c_list)) else 0.7,
                )
            k = k_active_list[i]
            ax[5].axvline(x= self.x_base_index[ k % n_index_basis, d0_index_set_plot ], c= c_list[i % len(c_list)], ls= '--', alpha=0.5)
        ax[5].set_ylabel(f'y')
        ax[5].set_xlabel(f'x')
        ax[5].set_title(f'included basis functions (sorted by beta)')
        ax[5].legend()


        ax[6].plot(
            sorted(np.log10(alpha_active_expect)), 
            marker='1',
            label= 'log10(alpha)',
            c = c_list[0], 
            )
        ax[6].axhline(
            y= np.log10(threshold_alpha), 
            label= f'threshold={threshold_alpha:.3e}', 
            c = c_list[1], 
            ls= '-.',
            alpha= 0.7,
            )
        ax[6].axhline(
            y= np.log10(self.max_precision_active_alpha), 
            label= f'max_precision_active={self.max_precision_active_alpha:.3e}', 
            c = c_list[2], 
            ls= '-.',
            alpha= 0.7,
            )
        idx_alpha_active_ineffective = np.where(
            np.logical_or(
                np.sort(alpha_active_expect) > threshold_alpha,
                np.isclose( np.sort(alpha_active_expect), threshold_alpha ),
            )
        )[0]
        if len( idx_alpha_active_ineffective )>0:
            ax[6].axvline(
                x= idx_alpha_active_ineffective.min(), label= f'thresholded', c = c_list[1], ls= '--',
                alpha= 0.7,
                )
        idx_alpha_active_bounded = np.where(
            np.logical_or(
                np.sort(alpha_active_expect) > self.max_precision_active_alpha,
                np.isclose( np.sort(alpha_active_expect), self.max_precision_active_alpha ),
            )
        )[0]
        if len( idx_alpha_active_bounded )>0:
            ax[6].axvline(
                x= idx_alpha_active_bounded.min(), label= f'bounded', c = c_list[2], ls= '--',
                alpha= 0.7,
                )
        ax[6].set_xlabel(f'j')
        ax[6].set_ylabel(f'log10(alpha)')
        ax[6].set_title(f'quantity of log10(alpha) (sorted). eta={(self.a_eta / self.b_eta):.3e}')
        ax[6].legend()

        ax[7].plot(
            sorted(np.log10(beta_active_expect)), 
            marker='1',
            label= 'log10(beta)',
            c = c_list[0], 
            )
        ax[7].axhline(
            y= np.log10(threshold_beta), 
            label= f'threshold={threshold_beta:.3e}', 
            c = c_list[1], 
            ls= '-.',
            alpha= 0.7,
            )
        ax[7].axhline(
            y= np.log10(self.max_precision_active_beta), 
            label= f'max_precision_active={self.max_precision_active_beta:.3e}', 
            c = c_list[2], 
            ls= '-.',
            alpha= 0.7,
            )
        idx_beta_active_ineffective = np.where(
            np.logical_or(
                np.sort(beta_active_expect) > threshold_beta,
                np.isclose( np.sort(beta_active_expect), threshold_beta ),
            )
        )[0]
        if len( idx_beta_active_ineffective )>0:
            ax[7].axvline(
                x= idx_beta_active_ineffective.min(), label= f'thresholded', c = c_list[1], ls= '--',
                alpha= 0.7,
                )
        idx_beta_active_bounded = np.where(
            np.logical_or(
                np.sort(beta_active_expect) > self.max_precision_active_beta,
                np.isclose( np.sort(beta_active_expect), self.max_precision_active_beta ),
            ) 
        )[0]
        if len( idx_beta_active_bounded )>0:
            ax[7].axvline(
                x= idx_beta_active_bounded.min(), label= f'bounded', c = c_list[2], ls= '--',
                alpha= 0.7,
                )
        ax[7].set_xlabel(f'k')
        ax[7].set_ylabel(f'log10(beta)')
        ax[7].set_title(f'quantity of log10(beta) (sorted)')
        ax[7].legend()


        if len(self.lower_bound_history_list)>0:
            ax[8].plot(
                self.lower_bound_history_list,
                label= 'lower_bound',
                )
            for i_n,n in enumerate(self.n_lbhist_new_basis_list):
                ax[8].axvline(x= n, label= f'add_new_basis' if i_n==0 else None, c = 'r', ls= '-.')
            for i_n,n in enumerate(self.n_lbhist_remove_basis_list):
                ax[8].axvline(x= n, label= f'remove_basis' if i_n==0 else None, c = 'g', ls= '-.')
    
            ax[8].legend()
    
        ax[8].set_title(f'lower bound plot, ending in {self.lower_bound_history_list[-1] if ( len(self.lower_bound_history_list)>0 and len(self.lower_bound_history_list)>0 ) else np.nan :.8e}')
        ax[8].set_xlabel(f'number of updates')
        ax[8].set_ylabel(f'lower bound')

        if len(self.lower_bound_history_list)>0:
            y = self.lower_bound_history_list[-n_recent:]
            x = np.arange( start= len(self.lower_bound_history_list) - len(y), stop=len(self.lower_bound_history_list), step= 1, dtype=int )
            ax[9].plot(
                x, 
                y,
                label= 'lower_bound',
                )

            n_lbhist_new_basis_list = np.array(self.n_lbhist_new_basis_list)
            for i_n,n in enumerate( n_lbhist_new_basis_list[ n_lbhist_new_basis_list>=x.min() ] ):
                ax[9].axvline(x= n, label= f'add_new_basis' if i_n==0 else None, c = 'r', ls= '-.')
            n_lbhist_remove_basis_list = np.array(self.n_lbhist_remove_basis_list)
            for i_n,n in enumerate( n_lbhist_remove_basis_list[ n_lbhist_remove_basis_list>=x.min() ] ):
                ax[9].axvline(x= n, label= f'remove_basis' if i_n==0 else None, c = 'g', ls= '-.')
    
            ax[9].legend()
    
        ax[9].set_title(f'lower bound plot (recent {len(x)}), ending in {self.lower_bound_history_list[-1] if ( len(self.lower_bound_history_list)>0 and len(self.lower_bound_history_list)>0 ) else np.nan :.8e}')
        ax[9].set_xlabel(f'number of updates')
        ax[9].set_ylabel(f'lower bound')


        if len(self.posterior_history_list)>0:
            ax[10].plot(self.posterior_history_list, label= 'posterior_probability')
            for i_n,n in enumerate(self.n_lbhist_new_basis_list):
                ax[10].axvline(x= n, label= f'add_new_basis' if i_n==0 else None, c = 'r', ls= '-.')
            for i_n,n in enumerate(self.n_lbhist_remove_basis_list):
                ax[10].axvline(x= n, label= f'remove_basis' if i_n==0 else None, c = 'g', ls= '-.')

            ax[10].legend()

        ax[10].set_title(f'posterior probability plot, ending in {self.posterior_history_list[-1] if ( len(self.posterior_history_list)>0 and len(self.posterior_history_list)>0 ) else np.nan :.8e}')
        ax[10].set_xlabel(f'number of updates')
        ax[10].set_ylabel(f'posterior probability')

        if len(self.posterior_history_list)>0:
            y = self.posterior_history_list[-n_recent:]
            x = np.arange( start= len(self.posterior_history_list) - len(y), stop=len(self.posterior_history_list), step= 1, dtype=int )
            ax[11].plot(x, y, label= 'posterior_probability')

            n_lbhist_new_basis_list = np.array(self.n_lbhist_new_basis_list)
            for i_n,n in enumerate( n_lbhist_new_basis_list[ n_lbhist_new_basis_list>=x.min() ] ):
                ax[11].axvline(x= n, label= f'add_new_basis' if i_n==0 else None, c = 'r', ls= '-.')
            n_lbhist_remove_basis_list = np.array(self.n_lbhist_remove_basis_list)
            for i_n,n in enumerate( n_lbhist_remove_basis_list[ n_lbhist_remove_basis_list>=x.min() ] ):
                ax[11].axvline(x= n, label= f'remove_basis' if i_n==0 else None, c = 'g', ls= '-.')
        
            ax[11].legend()
            
        ax[11].set_title(f'posterior probability plot (recent {len(x)}), ending in {self.posterior_history_list[-1] if ( len(self.posterior_history_list)>0 and len(self.posterior_history_list)>0 ) else np.nan :.8e}')
        ax[11].set_xlabel(f'number of updates')
        ax[11].set_ylabel(f'posterior probability')
        ax[11].legend()


        plt.tight_layout()

        if path_fig is not None:
            fig.savefig(path_fig)
        
        plt.close( fig )

        return mu_mise, cov_mise


    def fit(
        self,
        y_up_list,
        x_up_list,
        # shared
        sigma_init=None,
        sigma_is_fixed= False,
        n_log=1,
        sim_threshold_new_basis= 0.99,
        dir_debug=None,
        coef_func_index= None,

        # rvm
        ratio_init_var_noise= 0.1,
        n_iter_max_beta_init=100,
        n_patience_nrvm_vi=10,
        tol_converged_beta_init= None,
        tol_plateau_beta_init= None,
        n_basis_add_serial= 1,
        # init_iter_per_epoch= 1,
        nrvm_batch_size= None,
        nrvm_llkh = True,
        nrvm_cond_num_max= None,

        vi_or_gibbs= 'vi',
        refit_nrvm= True,
        sigma_is_fixed_rvm= False,

        # bsfda
        n_active_basis_init= None,
        n_iter_max= None,
        n_iter_max_delaywhite= 1000,
        tol_convergence_variational=None,
        # annealing
        use_pseudo = False,
        sigmak_init= 1e-2,
        sigmak_end= 1e-5,
        anneal_decay= 'lin',
        # init_sigmak_active= 1e-2,
        # aneal_sigmak_active = 1e-3,
        # init_corr_noise_ratio= 1e-2,
        # end_corr_noise_ratio= 1e-2,
        n_iter_anneal = None,
        tol_converge_rel_alpha_beta= 1e-7,
        n_iter_converge_plateau= 100,
        param2eval_err_rt = None,
        # test
        aba=None,
        remove_redundant_basis= True,
        ):
        '''
        y_up_list:
            list of n_subject observed functions stored as (1, n_up_i) arrays
        x_up_list:
            list of n_subject observed index set values stored as (1, n_up_i, d_index) arrays, d_index is the dimension of the index set
        n_iter_max_delaywhite:
            maximum iterations to hold off optimizing white noise, to get stable update of white noise
        remove_redundant_basis:
            whether to remove redundant basis functions during the optimization
        coef_func_index:
            coefficient for the size of index set for basis functions. To reduce computation with similar basis functions, the original index set is clustered into a smaller set. The size of the smaller set is coef_func_index times the largest number of measurements of sample functions.

        nrvm_llkh:
            whether to calculate the log likelihood of the data in the NRVM optimization to check convergence, False to speed up the optimization and run full iterations

        init_corr_noise_ratio:
            the ratio of the correlated noise to the signal coefficients, to set the initial value of sigmak
        n_iter_anneal:
            number of iterations to anneal the sigmak
        tol_converge_rel_alpha_beta:
            tolerrance for convergence based on relative change of alpha
        n_iter_converge_plateau:
            claim convergence if the change is small for this number of iterations
        refit_nrvm:
            whether to refit the NRVM or load the pre-fitted NRVM if available
        '''
        tol_converged_beta_init = 1e-5 * len(x_up_list) if tol_converged_beta_init is None else tol_converged_beta_init
        tol_convergence_variational = 1e-7 * len(x_up_list) if tol_convergence_variational is None else tol_convergence_variational
        n_iter_max = 20000 if n_iter_max is None else n_iter_max
        n_iter_anneal = int(n_iter_max/2) if n_iter_anneal is None else n_iter_anneal

        self.y_up_list = y_up_list
        self.x_up_list = x_up_list

        dtype = y_up_list[0].dtype
        self.max_precision_active_alpha = self.max_precision_active_alpha.astype(dtype)
        self.max_precision_active_beta = self.max_precision_active_beta.astype(dtype)
        self.precision_threshold_multiplier2min = self.precision_threshold_multiplier2min.astype(dtype)
        self.a0 = self.a0.astype(dtype)
        self.b0 = self.b0.astype(dtype)
        self.init_machine_precision = self.init_machine_precision.astype(dtype)
        self.cond_num_max = np.dtype(dtype).type(self.cond_num_max) if self.cond_num_max is not None else None

        self.EPS = self.EPS.astype(dtype)



        n_up_i_list = [
            t.shape[1] for t in self.y_up_list
        ]
        self.n_up_i_range = [np.min(n_up_i_list), np.max(n_up_i_list)]

        self.initialize_by_nrvm(
            y_up_list= y_up_list,
            x_up_list= x_up_list,
            dir_debug= dir_debug,

            sigma_init= sigma_init,
            ratio_init_var_noise= ratio_init_var_noise,
            sigma_is_fixed= sigma_is_fixed or sigma_is_fixed_rvm,

            # init_corr_noise_ratio= init_corr_noise_ratio,
            # end_corr_noise_ratio= end_corr_noise_ratio,
            sigmak_init= sigmak_init,
            sigmak_end= sigmak_end,

            n_iter_max_init= n_iter_max_beta_init,
            n_patience_fast_alpha= n_patience_nrvm_vi,
            tol_converged_beta_init= tol_converged_beta_init,
            sim_threshold = sim_threshold_new_basis,
            n_basis_add_serial= n_basis_add_serial,
            # iter_per_epoch= init_iter_per_epoch,
            batch_size= nrvm_batch_size,
            nrvm_llkh = nrvm_llkh,
            refit_nrvm= refit_nrvm,
            # em
            tol_plateau= tol_plateau_beta_init,
            coef_func_index= coef_func_index,

            n_active_basis_init= n_active_basis_init,
            cond_num_max= self.cond_num_max if nrvm_cond_num_max is None else nrvm_cond_num_max,

            n_log= max(int(n_log/100), 1),
        )
        if vi_or_gibbs == 'vi':
            self.variational_inference(
                y_up_list= y_up_list.copy(),

                init_zbar_active= self.init_z_bar_active,
                init_sigma_white= self.init_sigma_white,
                init_alpha_active= self.init_alpha_active.copy(),
                init_beta_active= self.init_beta_active.copy(),
                init_eta= self.init_eta,
                init_mu_vecwup_active=self.init_w_up_active.T.reshape(1,-1),
                init_mu_z_up_i_list_active= self.init_mu_z_up_i_list_active,

                # parameters specific to faster optimization
                init_mu_theta_i_list_active = self.init_mu_theta_i_list_active,
                init_sigmak_active = self.init_sigmak_active,
                aneal_sigmak_active = self.aneal_sigmak_active,
                n_iter_anneal = n_iter_anneal,
                use_pseudo_y = use_pseudo,
                anneal_decay= anneal_decay,
                tol_converge_rel_alpha_beta= tol_converge_rel_alpha_beta,
                n_iter_converge_plateau= n_iter_converge_plateau,

                update_white= not sigma_is_fixed,
                remove_redundant_basis = remove_redundant_basis,

                new_basis_corr_dataset_size= nrvm_batch_size,

                n_iter_max= n_iter_max,
                n_iter_max_delaywhite= n_iter_max_delaywhite,
                n_log= n_log,
                tol_convergence= tol_convergence_variational,
                sim_threshold_new_basis= sim_threshold_new_basis,
                n_patience= n_patience_nrvm_vi,
                param2eval_err_rt= param2eval_err_rt,
                dir_debug= dir_debug,
                # test
                aba=aba,
            )
        elif vi_or_gibbs == 'gibbs':
            # NOTE Gibbs sampling is legacy and not recommended
            self.gibbs_sample(
                y_up_list= y_up_list,

                init_zbar_active= self.init_z_bar_active,
                init_sigma_white= self.init_sigma_white,
                init_alpha_active= self.init_alpha_active.copy(),
                init_beta_active= self.init_beta_active.copy(),
                init_eta= self.init_eta,
                init_mu_vecwup_active=self.init_w_up_active.T.reshape(1,-1),
                init_mu_z_up_i_list_active= self.init_mu_z_up_i_list_active,

                # parameters specific to faster optimization
                init_mu_theta_i_list_active = self.init_mu_theta_i_list_active,
                init_sigmak_active = self.init_sigmak_active,

                update_white= not sigma_is_fixed,
                n_iter_max= n_iter_max,
                n_iter_max_delaywhite= n_iter_max_delaywhite,
                n_log= n_log,
                dir_debug= dir_debug,
            )

        logging.critical(f'BSFDA is fitted!')
        return self



    def gibbs_sample(
        self,
        y_up_list,
        init_zbar_active,
        init_sigma_white,
        init_alpha_active,
        init_beta_active,
        init_eta,
        init_mu_vecwup_active,
        init_mu_z_up_i_list_active,
        init_mu_theta_i_list_active,
        init_sigmak_active,
        update_white= True,
        n_iter_max= 1000,
        n_iter_max_delaywhite= 1000,
        n_log= 1,
        dir_debug= None,

        aba=None,


        burn_in= 0.1,
        thinning= 10,
        ):
        '''
        burn_in:
            proportion of burn in samples
        thinning:
            thinning of samples, only keep every thinning samples, if thinning=1, keep all samples, if thinning=2, keep every other samples, if thinning=3, keep every third samples, etc.
        '''

        n_index_basis = self.x_base_index.shape[0]
        k_active_list = self.k_active_list
        logging.debug(f'k_active_list= { k_active_list}')
        n_basis_active = len( k_active_list )

        z_bar_active = init_zbar_active.copy()
        sigma_white = init_sigma_white
        alpha_active = init_alpha_active.copy()
        beta_active = init_beta_active.copy()
        eta = init_eta
        vecwup_active = init_mu_vecwup_active.copy()
        w_up_active = vecwup_active.reshape((n_basis_active, n_basis_active)).T
        z_up_i_list_active = init_mu_z_up_i_list_active.copy()
        theta_i_list_active = init_mu_theta_i_list_active.copy()
        sigmak_active = init_sigmak_active.copy()

        a_white = np.sum([y.shape[1] for y in y_up_list]) / 2 + self.a0

        a_alpha_active = np.ones(shape=( n_basis_active )) * ( n_basis_active/2 ) + self.a0 if (aba is None) else np.ones(shape=( n_basis_active )) * aba

        a_eta = n_basis_active/2 + self.a0 if (aba is None) else aba

        a_beta_active = np.ones(shape=( n_basis_active )) * ( n_basis_active + 1 + len(y_up_list) )/2 + self.a0 if (aba is None) else np.ones(shape=( n_basis_active )) * aba

        # pre compute
        phi_up_list_active = [
            np.concatenate([
                self.kernel_list[ k // n_index_basis ]( self.x_base_index[ [k % n_index_basis] ], zi[0] )
                for k in self.k_active_list
            ], axis =0)
            for i, zi in enumerate(self.x_up_list)
        ]
        yyt_list = [
            yi@yi.T
            for i, yi in enumerate(y_up_list)
        ]
        yyt_list_sum = np.sum(yyt_list)
        ppt_list_active = [
            pi @ pi.T
            for i,pi in enumerate(phi_up_list_active)
        ]
        ppt_list_active_sum = (np.sum(ppt_list_active, axis=0))
        ppt_list_active_sum_inv = svd_inverse(
            ppt_list_active_sum, 
            cond_max=self.cond_num_max, 
            hermitian= True,
            logging_prefix = '[ppt_list_active_sum_inv] ',
            )
        ypt_list_active = [
            yi @ phi_up_list_active[i].T
            for i,yi in enumerate(y_up_list)
        ]
        ypt_list_active_sum = np.sum([
            ypt_list_active[i]
            for i,yi in enumerate(y_up_list)
        ], axis=0)


        var_nm2list = {
            'z_bar_active': [],
            'w_up_active': [],
            'alpha_active': [],
            'eta': [],
            'z_up_i_list_active': [],
            'theta_i_list_active': [],
            'beta_active': [],
            'sigmak_active': [],
            'sigma_white': [],
        }



        for i_iter in range(n_iter_max):

            

            # update z_bar
            z_bar_active_cov_inv = sigma_white**(-2) * ppt_list_active_sum 
            np.fill_diagonal(z_bar_active_cov_inv, z_bar_active_cov_inv.diagonal() + eta * sigmak_active)
            z_bar_active_cov = svd_inverse(
                z_bar_active_cov_inv, 
                cond_max=self.cond_num_max, 
                hermitian= True,
                logging_prefix = '[z_bar_active_cov_inv] ',
                )
            z_bar_active_mu = sigma_white**(-2) * ( ypt_list_active_sum - np.sum([
                ti @ ppt_list_active[i]
                for i, ti in enumerate(theta_i_list_active)
            ], axis=0) ) @ z_bar_active_cov
            # sample from the normal distribution
            z_bar_active = z_bar_active_mu + np.random.randn( z_bar_active_mu.shape[0], z_bar_active_mu.shape[1] ) @ np.linalg.cholesky( z_bar_active_cov ).T

            # update vecwup_active
            for k in range(n_basis_active):
                # update wup
                wup_k_cov_inv = beta_active[k]  * (
                    sigmak_active[k]**(-2) * np.sum([
                        zi.T @ zi
                        for zi in z_up_i_list_active
                    ], axis=0) 
                ) 
                np.fill_diagonal(wup_k_cov_inv, wup_k_cov_inv.diagonal() + beta_active[k] * alpha_active )
                wup_k_cov = svd_inverse(
                    wup_k_cov_inv, 
                    cond_max=self.cond_num_max, 
                    hermitian= True,
                    logging_prefix = f'[wup_k_cov_inv_{k}] ',
                    )
                wup_k_mu = sigmak_active[k]**(-2) * beta_active[k] * np.sum([
                    ti[0,k] * zi
                    for ti,zi in zip(theta_i_list_active, z_up_i_list_active)
                ], axis=0) @ wup_k_cov
                wup_k = wup_k_mu + np.random.randn( wup_k_mu.shape[0], wup_k_mu.shape[1] ) @ np.linalg.cholesky( wup_k_cov ).T

                w_up_active[:,[k]] = wup_k.T



            # update alpha
            for j in range( n_basis_active ):

                a_alpha_active_j = a_alpha_active[j]
                b_alpha_active_j = 1/2 * ( w_up_active[j,:]**2 * beta_active ).sum() + self.b0

                alpha_active[j] = np.random.gamma(shape=a_alpha_active_j, scale=1/b_alpha_active_j)

            # update eta
            b_eta = 1/2 * (z_bar_active**2 * beta_active).sum() + self.b0
            eta = np.random.gamma( shape= a_eta, scale= 1/b_eta)

            skn2b = sigmak_active**(-2) * beta_active

            # update z
            for i,zi in enumerate(z_up_i_list_active):
                zi_cov = svd_inverse(
                    w_up_active * ( skn2b )[None,:] @ w_up_active.T + np.identity(n= n_basis_active)
                )
                zi_mu = theta_i_list_active[i] * ( skn2b )[None,:] @ w_up_active.T @ zi_cov

                z_up_i_list_active[i] = zi_mu + np.random.randn( zi_mu.shape[0], zi_mu.shape[1] ) @ np.linalg.cholesky( zi_cov ).T

            # update theta
            for i,ti in enumerate(theta_i_list_active):
                t = ppt_list_active[i] * sigma_white**(-2)
                np.fill_diagonal(t, t.diagonal() + skn2b)
                ti_cov = svd_inverse(t)
                ti_mu = (
                    sigma_white**(-2) * ( y_up_list[i] @ phi_up_list_active[i].T - z_bar_active @ ppt_list_active[i] ) + z_bar_active @ w_up_active * ( skn2b )[None,:]
                ) @ ti_cov

                theta_i_list_active[i] = ti_mu + np.random.randn( ti_mu.shape[0], ti_mu.shape[1] ) @ np.linalg.cholesky( ti_cov ).T

            # update beta
            for k in range( n_basis_active ):
                a_beta_active_k = a_beta_active[k]
                b_beta_active_k = 1/2 * ( z_bar_active[0,k]**2 * eta + ( w_up_active[:,k]**2 * alpha_active ).sum() ) + sigmak_active[k]**(-2) * np.sum([
                    ( ti[0,k] - zi @ w_up_active[:,k] )**2
                    for ti,zi in zip(theta_i_list_active, z_up_i_list_active)
                ]) + self.b0

                beta_active[k] = np.random.gamma(shape=a_beta_active_k, scale=1/b_beta_active_k)

            # update sigmak
            # TODO
            pass

            # update sigma_white
            if update_white and (i_iter > n_iter_max_delaywhite):
                b_white = 1/2 * np.sum([
                    ( ( yi - ( z_bar_active + theta_i_list_active[i] ) @ phi_up_list_active[i] )**2 ).sum() 
                    for i,yi in enumerate(y_up_list)
                ]) + self.b0
                sigma_white = np.random.gamma( shape= a_white, scale= 1/b_white)**(-1/2)
            else:
                sigma_white

            # store the samples
            if (i_iter % thinning == 0) and (i_iter > burn_in * n_iter_max):
                var_nm2list['z_bar_active'].append( z_bar_active.copy() )
                var_nm2list['w_up_active'].append( w_up_active.copy() )
                var_nm2list['alpha_active'].append( alpha_active.copy() )
                var_nm2list['eta'].append( eta )
                var_nm2list['z_up_i_list_active'].append( z_up_i_list_active.copy() )
                var_nm2list['theta_i_list_active'].append( theta_i_list_active.copy() )
                var_nm2list['beta_active'].append( beta_active.copy() )
                var_nm2list['sigmak_active'].append( sigmak_active.copy() )
                var_nm2list['sigma_white'].append( sigma_white )

            if (i_iter % n_log == 0):
                logging.info(f'n_iter= {i_iter}, sigma= {sigma_white:.8e}, sigma_k= {sigmak_active}, \nalpha= {pprint.pformat(alpha_active, indent=4)}, \nalpha_zbar= {eta:.8e}, \nbeta= {pprint.pformat(beta_active, indent=4)}, \nmu_x= {z_bar_active}, \nmu_w= {w_up_active}')

        # calculate the statistics
        self.a_alpha_active = a_alpha_active
        self.b_alpha_active = np.zeros( shape=( n_basis_active ) )
        for j in range( n_basis_active ):
            self.b_alpha_active[j] = self.a_alpha_active[j] / np.array([a[j] for a in var_nm2list['alpha_active'] ]).mean()

        self.a_beta_active = a_beta_active
        self.b_beta_active = np.zeros( shape=( n_basis_active ) )
        for k in range( n_basis_active ):
            self.b_beta_active[k] = self.a_beta_active[k] / np.array([b[k] for b in var_nm2list['beta_active'] ]).mean()

        self.a_eta = a_eta
        self.b_eta = self.a_eta / np.array(var_nm2list['eta']).mean()

        self.a_white = a_white
        self.b_white = self.a_white / np.array(var_nm2list['sigma_white']).mean()

        self.mu_vecwup_active = np.array(var_nm2list['w_up_active']).mean(axis=0).T.reshape(1,-1)
        self.cov_vecwup_active = np.cov( np.array(var_nm2list['w_up_active']).transpose(0,2,1).reshape(len(var_nm2list['w_up_active']), -1).T )

        self.mu_zi_list_active = np.array(var_nm2list['z_up_i_list_active']).mean(axis=0)
        z_up_i_list_active_sample = np.array(var_nm2list['z_up_i_list_active'])
        self.cov_zi_list_active = [
            np.cov( z_up_i_list_active_sample[:,i,:,:].squeeze(axis=1).T )
            for i in range(z_up_i_list_active_sample.shape[1])
        ]

        self.mu_zbar_active = np.array(var_nm2list['z_bar_active']).mean(axis=0)
        self.cov_zbar_active = np.cov( np.array(var_nm2list['z_bar_active']).squeeze( axis=1 ).T )

        self.mu_theta_i_list_active = np.array(var_nm2list['theta_i_list_active']).mean(axis=0)
        theta_i_list_active_sample = np.array(var_nm2list['theta_i_list_active'])
        self.cov_theta_i_list_active = [
            np.cov( theta_i_list_active_sample[:,i,:,:].squeeze(axis=1).T )
            for i in range(theta_i_list_active_sample.shape[1])
        ]


        return self



                


            

    def try_activate_new_beta(
        self,

        mu_zi_list_active,
        cov_zi_list_active,
        mu_vecwup_active,
        cov_vecwup_active,
        mu_zbar_active,
        cov_zbar_active,
        alpha_active,
        beta_active,
        sigma_white,

        eta,

        mu_theta_i_list_active,
        cov_theta_i_list_active,

        sigmak_active,

        phi_up_list_active,

        update_white = True,
        update_vecwup = True,
        update_z_up = True,
        update_zbar = True,
        update_alpha = True,
        update_eta = True,
        update_theta = True,

        update_beta= True,

        tol_converge_rel_alpha_beta= 1e-7,
        n_iter_converge_plateau= 100,

        use_pseudo_y = False,

        n_iter_max= 1,
        n_iter_max_delaywhite= 0,
        n_log= 1,
        tol_convergence= None,
        dir_debug= None,
        sim_threshold = 0.999,
        new_basis_corr_dataset_size= None,

        KernelPresLatLonTime_threshold = KernelPresLatLonTime_threshold,
    ):
        '''
        try activing beta one by one and return the best one
        ---
        new_basis_corr_dataset_size:
            size of the dataset to calculate the correlation with the new basis function, None to use the full dataset
        '''
        corr_idx = slice(None) if new_basis_corr_dataset_size is None else slice(new_basis_corr_dataset_size)

        if (dir_debug is not None) and (not os.path.exists(dir_debug)):
            os.makedirs( dir_debug )

        tol_convergence = 1e-8 * len(self.y_up_list) if (tol_convergence is None) else tol_convergence

        k_active_list = self.k_active_list

        threshold_beta = min( (beta_active.min() * self.precision_threshold_multiplier2min), self.max_precision_active_beta )

        # reset the list of newly added basis functions if there is one that has gained substantial weights
        logging.info(f'self.k_list_newly_tried= {self.k_list_newly_tried}')
        for k in list(self.k_list_newly_tried):
            if k in k_active_list:
                k_idx = k_active_list.index( k )
                if beta_active[ k_idx ] < threshold_beta:
                    logging.info(f'resetting self.k_list_newly_tried')
                    self.k_list_newly_tried = []
                    break

        n_index_set = self.x_base_index.shape[0]
        n_basis = n_index_set * len(self.kernel_list)
        n_basis_active = len(k_active_list)

        w_up_active = mu_vecwup_active.reshape((-1, n_basis_active)).T

        # find the basis functions that are most unlike to those active
        k_try_list = []
        # sort basis functions by similarity to th residues
        k_list = list(range( n_basis ))
        y_up_list_residue = [
            yi[:,corr_idx] - mu_zi_list_active[i] @ w_up_active @ phi_up_list_active[i][:,corr_idx]
            for i,yi in enumerate(self.y_up_list)
        ]
        # sort by correlation with the residues
        k2inner_product_sum = {}
        for k in k_list:
            if (k in k_active_list) or (k in self.k_list_similar2active):
                continue
            p_try_list = [
                self.kernel_list[ k//n_index_set ]( self.x_base_index[ [k % n_index_set] ], self.x_up_list[i][0][corr_idx] ).flatten()
                for i,yi in enumerate(self.y_up_list)
            ]
            k2inner_product_sum[k] = np.sum([
                np.abs( y_up_list_residue[i] @ p_try_list[i].T )
                for i,yi in enumerate(self.y_up_list)
            ])
        k2inner_product_sum = sorted( list(k2inner_product_sum.items()), key= lambda e:e[1], reverse= True)
        logging.debug(f'k2inner_product_sum= {pprint.pformat(k2inner_product_sum, indent= 4)}')
        k_list_sorted = np.array(k2inner_product_sum)[:,0].astype(int).tolist() if ( len(k2inner_product_sum)>0 ) else []
        
        # NOTE will skip k_list_newly_tried
        fm_active = self.k_list_get_fm( k_list= sorted(list(set(k_active_list + self.k_list_newly_tried + k_try_list))) )
        # filter by thresholding similarity
        for k_try in k_list_sorted:
            # cosine similarity between the new basis function and the optimized ones
            # reuse fm_active to save time from orthornomalization

            is_not_similar = None
            if isinstance( self.kernel_list[0], RBFKernel2 ):
                sim = fm_active.cosine_similar2subspace(
                    rbfm= self.k_list_get_fm( k_list= [k_try,] ),
                )
                is_not_similar = sim < sim_threshold
                logging.debug(f'sim= {sim} for k_try= {k_try}')
            elif isinstance( self.kernel_list[0], KernelPresLatLonTime ):
                is_not_similar = not fm_active.check_if_similar( self.k_list_get_fm( k_list= [k_try,] ).lat_lon_pres_time_function_list[0], threshold= KernelPresLatLonTime_threshold )
            else:
                raise NotImplementedError(f'kernel type not implemented yet')

            if is_not_similar:
                
                k_try_list.append( k_try )

                # NOTE only try the one with the highest residue.
                break
                
        logging.debug(f'k_try_list= {k_try_list}')
        self.k_list_newly_tried += k_try_list
        dtype = mu_zi_list_active[0].dtype
        k2beta = {}
        optimal_k_self= None
        # lower_bound_init = np.nan
        for i_try, k_try in enumerate(k_try_list):

            self_copy = BSFDA(kernel_list= self.kernel_list)
            self_copy.__dict__.update( copy.deepcopy( self.__dict__ ) )

            # NOTE functions will be replaced as long as their precision is ineffective even if they are not max_precision_active_beta. this avoids inefficiency with many ineffective functions. being ineffective at convergence implies that they are not important and can be replaced. though precision of the new function will still start from max_precision_active_beta.
            k_active_idx_large = np.where( beta_active >= threshold_beta)[0]
            if len(k_active_idx_large)>0:

                logging.info(f'it will replace the large beta precision')

                k_active_idx_new = k_active_idx_large[0]
                
                self_copy.k_active_list[k_active_idx_new] = k_try

                alpha_active_tmp = alpha_active.copy()
                beta_active_tmp= beta_active.copy()
                beta_active_tmp[ k_active_idx_new ] = self.max_precision_active_beta

                mu_x_active_tmp= mu_zbar_active.copy()
                cov_x_active_tmp= cov_zbar_active.copy()

                mu_vecwup_active_tmp= mu_vecwup_active.copy()
                cov_vecwup_active_tmp= cov_vecwup_active.copy()

                mu_xi_list_active_tmp= mu_zi_list_active.copy()
                cov_xi_list_active_tmp= cov_zi_list_active.copy()

                mu_theta_i_list_active_tmp= mu_theta_i_list_active.copy()
                cov_theta_i_list_active_tmp= cov_theta_i_list_active.copy()

            else:
                logging.info(f'it will add a new beta precision')

                k_active_idx_new = len(self_copy.k_active_list)

                self_copy.k_active_list.append(k_try)

                alpha_active_tmp = alpha_active.copy()
                alpha_active_tmp = np.insert( alpha_active_tmp, n_basis_active, self.max_precision_active_beta )
                
                beta_active_tmp = beta_active.copy()
                beta_active_tmp = np.insert( beta_active_tmp, n_basis_active, self.max_precision_active_beta )

                mu_x_active_tmp = np.insert( mu_zbar_active, obj=n_basis_active, values= self.init_machine_precision, axis=1 )
                cov_x_active_tmp = np.identity(n= n_basis_active+1, dtype=dtype ) * self.init_machine_precision
                cov_x_active_tmp[ :k_active_idx_new, :k_active_idx_new ] = cov_zbar_active

                w_up_active_tmp = np.ones(shape=(n_basis_active+1, n_basis_active+1), dtype=dtype) * self.init_machine_precision
                w_up_active_tmp[ :k_active_idx_new, :k_active_idx_new ] = w_up_active
                mu_vecwup_active_tmp = w_up_active_tmp.T.reshape((1,-1))

                cov_vecwup_active_tmp = np.identity( n= (n_basis_active+1)**2, dtype=dtype ) * self.init_machine_precision
                k_active_list_wupvec_tmp = []
                for j in range(n_basis_active):
                    for k in range(n_basis_active):
                        k_active_list_wupvec_tmp.append( k*(n_basis_active+1) + j )
                cov_vecwup_active_tmp[ np.ix_( k_active_list_wupvec_tmp, k_active_list_wupvec_tmp ) ] = cov_vecwup_active

                mu_xi_list_active_tmp = [np.insert( m, obj= n_basis_active, values= self.init_machine_precision, axis= 1 ) for m in mu_zi_list_active]
                cov_xi_list_active_tmp = [ np.identity(n= n_basis_active+1, dtype=dtype ) * self.init_machine_precision for c in cov_zi_list_active ]
                for i, c in enumerate(cov_zi_list_active):
                    cov_xi_list_active_tmp[i][ :n_basis_active, :n_basis_active] = c

                mu_theta_i_list_active_tmp = [np.insert( m, obj= n_basis_active, values= self.init_machine_precision, axis= 1 ) for m in mu_theta_i_list_active]
                cov_theta_i_list_active_tmp = [ np.identity(n= n_basis_active+1, dtype=dtype ) * self.init_machine_precision for c in cov_theta_i_list_active ]
                for i, c in enumerate(cov_theta_i_list_active):
                    cov_theta_i_list_active_tmp[i][ :n_basis_active, :n_basis_active] = c


            self_copy.variational_inference(
                y_up_list= self.y_up_list.copy(),

                init_zbar_active= mu_x_active_tmp,
                init_cov_zbar_active= cov_x_active_tmp,
                
                init_sigma_white= sigma_white,
                init_alpha_active = alpha_active_tmp,
                init_beta_active= beta_active_tmp,

                init_eta = eta,

                init_mu_vecwup_active= mu_vecwup_active_tmp,
                init_cov_vecwup_active= cov_vecwup_active_tmp,

                init_mu_z_up_i_list_active= mu_xi_list_active_tmp,
                init_cov_z_up_i_list_active= cov_xi_list_active_tmp,

                update_beta= update_beta,

                update_white = update_white,
                update_vecwup = update_vecwup,
                update_z_up = update_z_up,
                update_zbar = update_zbar,
                update_alpha = update_alpha,
                update_eta = update_eta,
                update_theta = update_theta,
                # keep the same basis functions
                remove_redundant_basis= False,

                # parameters specific to the fast fitting
                init_mu_theta_i_list_active = mu_theta_i_list_active_tmp,
                init_cov_theta_i_list_active = cov_theta_i_list_active_tmp,
                
                init_sigmak_active = sigmak_active.flatten()[0],
                aneal_sigmak_active = sigmak_active.flatten()[0],
                sigmak_fix_iter = 1,
                n_iter_anneal = 2,
                use_pseudo_y = use_pseudo_y,
                tol_converge_rel_alpha_beta= tol_converge_rel_alpha_beta,
                n_iter_converge_plateau= n_iter_converge_plateau,

                # NOTE do not add any new basis functions
                sim_threshold_new_basis= -1,

                n_iter_max= n_iter_max,
                n_iter_max_delaywhite= n_iter_max_delaywhite,
                # NOTE no need to wait for convergence
                n_patience= n_iter_max,
                n_log= n_log,
                
                tol_convergence= tol_convergence,
                dir_debug= None if (dir_debug is None) else f'{dir_debug}/{i_try}_k{k_try}_ip{dict(k2inner_product_sum)[k_try]:.3e}' ,
            )
            
            # if lower_bound_init is np.nan:
            #     lower_bound_init = self_copy.lower_bound_history_list[0]
            #     logging.debug(f'lower_bound_init = {lower_bound_init}')

            beta_try = self_copy.a_beta_active[k_active_idx_new] / self_copy.b_beta_active[k_active_idx_new]

            logging.debug(f'beta_try= {beta_try}, max_precision_active_beta= {self.max_precision_active_beta}')

            # the added precision decreases
            if ( beta_try < self.max_precision_active_beta ):

                if (len(k2beta)==0) or (beta_try < np.array(list(k2beta.values())).min()):
                    optimal_k_self = copy.deepcopy(self_copy)

            k2beta[k_try] = beta_try

            logging.debug(f'k2beta= {pprint.pformat(sorted(list(k2beta.items()), key=lambda e:e[1], reverse=False), indent=4)}')


        if ( optimal_k_self is not None ):

            logging.debug(f'more bases will be added')

            k2beta_sorted = sorted(list(k2beta.items()), key=lambda e:e[1], reverse=False)

            # replace all the inactive
            k_active_idx_large = np.where( optimal_k_self.a_beta_active/ optimal_k_self.b_beta_active >= self.max_precision_active_beta )[0]

            if len(k_active_idx_large)>0:

                logging.info(f'there are more than one effective basis functions. it will use them to replace the large beta precision in the current model')

                for ik,(k_try,v) in enumerate( k2beta_sorted[1:] ):

                    if (ik >= len(k_active_idx_large)) or (v>= self.max_precision_active_beta):

                        break
                    
                    else:

                        k_active_idx_new = k_active_idx_large[ik]

                        optimal_k_self.k_active_list[ k_active_idx_new ] = k_try

        else:
        
            logging.debug(f'no more basis function will be added. ')
        
        return optimal_k_self
            


    def initialize_by_nrvm(
        self,

        y_up_list,
        x_up_list,

        n_iter_max_init=100,
        n_patience_fast_alpha= 10,
        tol_converged_beta_init=1e-6,

        sim_threshold = 0.999,
        n_basis_add_serial= 1,

        tol_plateau= None,
        nrvm_llkh = True,

        coef_func_index= None,
          
        dir_debug = None,

        sigma_init=None,
        ratio_init_var_noise= 1e-1,
        sigma_is_fixed= False,
        # iter_per_epoch= 1,
        batch_size= None,
        refit_nrvm= True,
        cond_num_max= None,

        # init_corr_noise_ratio= 5e-1,
        # end_corr_noise_ratio= 1e-2,
        sigmak_init= 1e-2,
        sigmak_end= 1e-2,

        n_log=1,

        n_active_basis_init= None,
        ):
        '''
        compute the expetation of beta
        '''
        dtype = y_up_list[0].dtype
        path_nrvm = f'{dir_debug}/nrvm.pkl'
        if ( (not refit_nrvm) and os.path.exists(path_nrvm) ):
            logging.info(f'loading nrvm from {path_nrvm}')
            with open(path_nrvm, 'rb') as f:
                nrvm = dill.load(f)
            # update the data
            nrvm.x_up_list = x_up_list
            nrvm.y_up_list = y_up_list
            # nrvm.phi_xi_list = [
            #     nrvm.phi(xi[0,:], x_base_index=nrvm.x_base_index) 
            #     for xi in x_up_list
            # ]

        else:
            logging.info(f'fitting nrvm')
            nrvm = nRVM(
                kernel_list= self.kernel_list, 
                max_precision_active = self.EPS**(-1),
                EPS= self.EPS,
                cond_num_max= cond_num_max,
                rcond_pinv= self.rcond_pinv,
                )
            nrvm.fit(
                x_up_list= x_up_list.copy(),
                y_up_list= y_up_list.copy(),
                # initialization
                coef_func_index= coef_func_index,
                sigma_init= sigma_init,
                ratio_init_var_noise= ratio_init_var_noise,
                # fast fitting
                sigma_is_fixed= sigma_is_fixed,
                n_iter_max_fast= n_iter_max_init,
                n_patience= n_patience_fast_alpha,
                tol= tol_converged_beta_init,
                sim_threshold = sim_threshold,
                n_basis_add_serial= n_basis_add_serial,
                # iter_per_epoch= iter_per_epoch,
                batch_size= batch_size,
                compute_loglike= nrvm_llkh,
                # em update
                tol_plateau= tol_plateau,
                # trim
                n_log= n_log,
                dir_debug = dir_debug,
            )
            if dir_debug is not None:
                nrvm.plot_results(
                    z_bar=nrvm.z_bar,        
                    beta=nrvm.beta,
                    sigma=nrvm.sigma,
                    path_fig = f'{dir_debug}/tmp.pdf',
                )

        j2j_active_alpha= {}
        nrvm_beta_min = nrvm.beta.min()
        max_precision_active_dynamic_beta_ = nrvm.get_max_precision_active_dynamic_beta(nrvm.beta)
        # NOTE this is because the precision will be scaled such that the min is 1
        for j,aj in enumerate(nrvm.beta):
            if aj < max_precision_active_dynamic_beta_:
                j2j_active_alpha[j] = len(j2j_active_alpha)
        k_active_list_nvm = sorted(list(j2j_active_alpha.keys()))

        n_index_basis = nrvm.x_base_index.shape[0]
        phi_up_list_active = [
            np.concatenate([
                nrvm.kernel_list[ k // n_index_basis ]( nrvm.x_base_index[ [k % n_index_basis] ], xi[0] )
                for k in k_active_list_nvm
            ], axis =0)
            for i, xi in enumerate(x_up_list)
        ] if (not hasattr(nrvm, 'phi_xi_list')) else None

        z_i_e_list_active, ztz_i_e_list_active, z_i_cov_list_active = nrvm.get_update_z_e(
            z_bar=nrvm.z_bar,        
            beta=nrvm.beta,
            sigma=nrvm.sigma,
            active_only= True,
            phi_xi_list_active= phi_up_list_active,
            k_active_beta= k_active_list_nvm,
        )

        # set initial values
        # NOTE fpca and rvm share the basis functions
        self.x_base_index = nrvm.x_base_index.copy()
        n_basis = self.x_base_index.shape[0]

        self.k_active_list = []
        idx_beta_active_bounded = []
        if n_active_basis_init is not None:
            if len(k_active_list_nvm)>=n_active_basis_init:
                k_active_list_nvm = k_active_list_nvm[:n_active_basis_init]
            else:
                # farthest point sampling, to avoid singular matrix
                while( len(k_active_list_nvm) < n_active_basis_init ):
                    kd = np.abs( np.arange(len(nrvm.beta))[:,None] - np.array(k_active_list_nvm)[None,:] ).min( axis=1 )
                    k = int(np.argmax(kd))

                    idx_beta_active_bounded.append( len(k_active_list_nvm) )
                    k_active_list_nvm.append(k)

        for k in k_active_list_nvm:
            n_k = k // (nrvm.x_base_index.shape[0])
            k_offset = k % (nrvm.x_base_index.shape[0])
            self.k_active_list.append( n_k * n_basis + np.where( np.all( self.x_base_index == nrvm.x_base_index[k_offset], axis=1 ) )[0][0] )

        self.k_list_similar2active = copy.deepcopy(nrvm.j_list_similar2active)

        self.init_beta_active = nrvm.beta[ k_active_list_nvm ].copy()
        
        self.init_alpha_active = np.ones_like(self.init_beta_active)

        self.init_eta = np.dtype(dtype).type( 1 )

        self.init_z_bar_active = nrvm.z_bar[:, k_active_list_nvm].copy()

        self.init_sigma_white = nrvm.sigma

        self.init_w_up_active = np.diag(self.init_beta_active**(-0.5))

        self.init_mu_z_up_i_list_active = [
            ( ( z_i_e_list_active[i] ) / nrvm.beta[ k_active_list_nvm ][None,:]**(-0.5) )
            for i,zi in enumerate(z_i_e_list_active)
        ]

        for k in idx_beta_active_bounded:
            self.init_alpha_active[k] = self.max_precision_active_beta
            self.init_z_bar_active[0,k] = self.init_machine_precision
            for i,zi in enumerate(self.init_mu_z_up_i_list_active):
                self.init_mu_z_up_i_list_active[i][0,k] = self.init_machine_precision


        # NOTE unify the smallest precision for numerical stability
        self.init_beta_active = self.init_beta_active / nrvm_beta_min
        for ik, k in enumerate( self.kernel_list ):
            self.kernel_list[ik].multiplier /= nrvm_beta_min**0.5
        self.init_w_up_active = self.init_w_up_active * nrvm_beta_min**0.5
        self.init_z_bar_active = self.init_z_bar_active * nrvm_beta_min**0.5

        # parameters specific to the fast fitting
        self.init_mu_theta_i_list_active =  [
            self.init_mu_z_up_i_list_active[i] @ self.init_w_up_active
            for i in range(len(self.init_mu_z_up_i_list_active))
        ]
        # HACK sigmak is shared
        # init_sigmak_active = init_corr_noise_ratio * self.init_beta_active.max()**(-0.5)
        init_sigmak_active = sigmak_init
        self.init_sigmak_active = np.ones(shape=( len(self.k_active_list) ), dtype= dtype) * init_sigmak_active
        logging.info(f'init_sigmak_active= {self.init_sigmak_active}')
        # self.aneal_sigmak_active = self.init_sigmak_active * (end_corr_noise_ratio/init_corr_noise_ratio)
        self.aneal_sigmak_active = np.dtype(dtype).type( sigmak_end )
        logging.info(f'aneal_sigmak_active= {self.aneal_sigmak_active}')


        return 


    def k_list_get_fm(
        self,
        k_list,
        ):
        n_index_bases = len(self.x_base_index)

        # if self.kernel_list[0] is RadialBasisFunction:
        if isinstance( self.kernel_list[0], RBFKernel2 ):
            
            f_j_list = [
                RadialBasisFunction(
                    length_scale= self.kernel_list[ j // n_index_bases ].length_scale,
                    center= self.x_base_index[ [j % n_index_bases] ],
                    multiplier= self.kernel_list[ j // n_index_bases ].multiplier,
                    )
                for j in k_list
            ]

            fm_j = RadicalBasisFunctionMixture( radial_basis_function_list= f_j_list )

        elif isinstance( self.kernel_list[0], KernelPresLatLonTime ):

            f_j_list = [
                PresLatLonTimeFunction(
                    sigma= self.kernel_list[ j // n_index_bases ].sigma,
                    length_scale_geo = self.kernel_list[ j // n_index_bases ].length_scale_geo, 
                    length_scale_pres = self.kernel_list[ j // n_index_bases ].length_scale_pres, 
                    length_scale_time = self.kernel_list[ j // n_index_bases ].length_scale_time, 
                    period_time = self.kernel_list[ j // n_index_bases ].period_time,
                    multiplier= self.kernel_list[ j // n_index_bases ].multiplier,
                    center= self.x_base_index[ [j % n_index_bases] ],
                    )
                for j in k_list
            ]

            fm_j = PresLatLonTimeFunctionMixture( lat_lon_pres_time_function_list= f_j_list )

        return fm_j

    def variational_inference(
        self,

        y_up_list,

        n_iter_max=100,
        n_iter_max_delaywhite=1,

        # initial status
        init_sigma_white = None,
        update_white = True,

        init_mu_vecwup_active = None,
        init_cov_vecwup_active = None,
        update_vecwup = True,

        init_mu_z_up_i_list_active = None,
        init_cov_z_up_i_list_active = None,
        update_z_up = True,

        init_zbar_active = None,
        init_cov_zbar_active = None,
        update_zbar = True,

        init_alpha_active = None,
        update_alpha = True,

        update_eta= True,

        init_eta = None,

        init_beta_active = None,
        update_beta = True,
        remove_redundant_basis = True,

        # parameters specific to the fast fitting
        init_mu_theta_i_list_active = None,
        init_cov_theta_i_list_active = None,
        update_theta = True,
        use_pseudo_y = False,
        anneal_decay = 'lin',
        
        init_sigmak_active = None,
        aneal_sigmak_active = None,
        sigmak_fix_iter = None,
        n_iter_anneal = int(1e3),
        tol_converge_rel_alpha_beta= 1e-7,
        n_iter_converge_plateau= 100,

        param2eval_err_rt= None,
        n_log = 1,

        new_basis_corr_dataset_size= None,

        tol_convergence = None,

        sim_threshold_new_basis= 0.999,

        n_patience = 10,
        
        dir_debug= None,

        aba= None,
        ):
        '''
        n_patience:
            the number of iterations to wait for the convergence of the lower bound
        '''
        if (dir_debug is not None) and (not os.path.exists(dir_debug)):
            os.makedirs( dir_debug )

        if sigmak_fix_iter is None:
            sigmak_fix_iter = int(n_iter_anneal/2)
        if sigmak_fix_iter > n_iter_max:
            sigmak_fix_iter = n_iter_max-1
        dtype = y_up_list[0].dtype
        tol_convergence = 1e-8 * len(y_up_list) if (tol_convergence is None) else tol_convergence

        self.lower_bound_history_list=[]
        self.posterior_history_list=[]

        n_index_basis = self.x_base_index.shape[0]

        k_active_list = self.k_active_list
        logging.debug(f'k_active_list= { k_active_list}')
        n_basis_active = len( k_active_list )

        n_iter_max_delaywhite_ = n_iter_max_delaywhite
        n_patience_countdown  = n_patience

        # initialize
        a_white = np.sum([y.shape[1] for y in y_up_list]) / 2 + self.a0
        b_white = a_white if init_sigma_white is None else a_white/init_sigma_white**(-2)

        mu_vecwup_active = init_mu_vecwup_active.copy()
        cov_vecwup_active = np.identity(n= n_basis_active * n_basis_active, dtype=dtype) * self.init_machine_precision if (init_cov_vecwup_active is None) else init_cov_vecwup_active.copy()

        if init_mu_z_up_i_list_active is not None:
            mu_zi_list_active = [e.copy() for e in init_mu_z_up_i_list_active]
        else:
            mu_zi_list_active = [
                np.ones( shape= ( 1, n_basis_active ), dtype=dtype) * self.init_machine_precision
                for i,yi in enumerate(y_up_list)
            ] 

        if init_cov_z_up_i_list_active is not None:
            cov_zi_list_active = [e.copy() for e in init_cov_z_up_i_list_active]
        else:
            cov_zi_list_active = [
                np.identity( n = n_basis_active, dtype=dtype ) * self.init_machine_precision
                for i,yi in enumerate(y_up_list)
            ] 

        if init_zbar_active is not None:
            mu_zbar_active = init_zbar_active.copy()
        else:
            mu_zbar_active = np.ones(shape=(1,n_basis_active), dtype=dtype) * self.init_machine_precision

        if init_cov_zbar_active is None:
            cov_zbar_active = np.identity(n= n_basis_active, dtype=dtype) * self.init_machine_precision
        else:
            cov_zbar_active = init_cov_zbar_active.copy()

        a_alpha_active = np.ones(shape=( n_basis_active ), dtype=dtype) * ( n_basis_active/2 ) + self.a0 if (aba is None) else np.ones(shape=( n_basis_active ), dtype=dtype) * aba
        if init_alpha_active is not None:
            b_alpha_active = (a_alpha_active / init_alpha_active).copy()
        else:
            b_alpha_active = a_alpha_active / self.max_precision_active_alpha

        a_eta = n_basis_active/2 + self.a0 if (aba is None) else aba
        if init_eta is not None:
            b_eta = a_eta / init_eta
        else:
            b_eta = a_eta / self.max_precision_active_alpha


        a_beta_active = np.ones(shape=( n_basis_active ), dtype=dtype) * ( n_basis_active + 1 + len(y_up_list) )/2 + self.a0 if (aba is None) else np.ones(shape=( n_basis_active ), dtype=dtype) * aba
        if init_beta_active is not None:
            b_beta_active = a_beta_active / init_beta_active
        else:
            b_beta_active = a_beta_active / self.max_precision_active_beta

        # parameters specific to the fast fitting
        if init_mu_theta_i_list_active is not None:
            mu_theta_i_list_active = [e.copy() for e in init_mu_theta_i_list_active]
        else:
            mu_theta_i_list_active = [
                np.ones( shape= ( 1, n_basis_active ), dtype=dtype) * self.init_machine_precision
                for i,yi in enumerate(y_up_list)
            ]
        if init_cov_theta_i_list_active is not None:
            cov_theta_i_list_active = [e.copy() for e in init_cov_theta_i_list_active]
        else:
            cov_theta_i_list_active = [
                np.identity( n = n_basis_active, dtype=dtype ) * self.init_machine_precision
                for i,yi in enumerate(y_up_list)
            ]
        
        a_sigmak_active = np.ones(shape=( n_basis_active ), dtype=dtype) * (len(y_up_list)/2) + self.a0
        if init_sigmak_active is not None:
            b_sigmak_active = a_sigmak_active / init_sigmak_active**(-2)
        else:
            b_sigmak_active = a_sigmak_active / self.max_precision_active_alpha
        b_sigmak0_active_it0 = b_sigmak_active.copy()[0]

        # original gap between the smallest alpha and beta, it should not be increased during the optimization
        # NOTE this is a hyperparameter
        max_log10_gap_min_alpha_beta = 1


        # pre compute
        phi_up_list_active = [
            np.concatenate([
                self.kernel_list[ k // n_index_basis ]( self.x_base_index[ [k % n_index_basis] ], zi[0] )
                for k in self.k_active_list
            ], axis =0)
            for i, zi in enumerate(self.x_up_list)
        ]
        yyt_list = [
            yi@yi.T
            for i, yi in enumerate(y_up_list)
        ]
        yyt_list_sum = np.sum(yyt_list)
        ppt_list_active = [
            pi @ pi.T
            for i,pi in enumerate(phi_up_list_active)
        ]
        ppt_list_active_sum = (np.sum(ppt_list_active, axis=0))
        # ppt_list_active_sum_inv = svd_inverse(
        #     ppt_list_active_sum, 
        #     cond_max=self.cond_num_max, 
        #     hermitian= True,
        #     logging_prefix = '[ppt_list_active_sum_inv] ',
        #     )
        ypt_list_active = [
            yi @ phi_up_list_active[i].T
            for i,yi in enumerate(y_up_list)
        ]
        ypt_list_active_sum = np.sum([
            ypt_list_active[i]
            for i,yi in enumerate(y_up_list)
        ], axis=0)

        # optimization loop
        n_iter = 0
        converged = False
        lower_bound_old = -np.inf
        start_time = time.time()
        start_resources = resource.getrusage(resource.RUSAGE_SELF)
        iter_time_utime_npc_mucov_error_list = []

        mu_theta_i_list_active_relachange = []
        alpha_active_relachange = []
        beta_active_relachange = []
        theta_minus_zw_norm_list = []

        n_iter_converge_plateau_cntdw = n_iter_converge_plateau
        while( (not converged) and (n_iter< n_iter_max) ):

            if ( aneal_sigmak_active is not None):
                b_sigmak_active_end = a_sigmak_active / aneal_sigmak_active**(-2)
                b_sigmak_active_it0 = np.ones_like(b_sigmak_active) * b_sigmak0_active_it0
                if ( n_iter > n_iter_anneal ): 
                    b_sigmak_active = b_sigmak_active_end
                else:
                    if ( n_iter < sigmak_fix_iter ):
                        b_sigmak_active = b_sigmak_active_it0
                    else:
                        if anneal_decay == 'exp':
                            # exponential annealing
                            b_sigmak_active = b_sigmak_active_it0 * np.exp( ( (n_iter-sigmak_fix_iter)/(n_iter_anneal-sigmak_fix_iter)) * np.log( b_sigmak_active_end / b_sigmak_active_it0 ) )
                        elif anneal_decay == 'lin':
                            # linear annealing
                            b_sigmak_active = b_sigmak_active_it0 + (n_iter-sigmak_fix_iter)/(n_iter_anneal-sigmak_fix_iter) * (b_sigmak_active_end - b_sigmak_active_it0)
                        elif anneal_decay == 'log':
                            # logaritmic annealing, Nourani, Yaghout, and Bjarne Andresen. "A comparison of simulated annealing cooling strategies." Journal of Physics A: Mathematical and General 31, no. 41 (1998): 8373.
                            a_ = ( b_sigmak_active_it0 * np.log(2) )
                            b_ = ( np.exp( a_ / b_sigmak_active_end ) - 2 ) / (n_iter_anneal-sigmak_fix_iter)
                            b_sigmak_active = a_ / np.log( b_ * (n_iter-sigmak_fix_iter) + 2 )


            elif ( n_iter > n_iter_max ):
                break
            
            
            # modify y_up_list to align with sigmak_active
            if use_pseudo_y:
                beta_ = a_beta_active / b_beta_active
                sigmak_active_ = (a_sigmak_active / b_sigmak_active)**(-0.5)
                y_up_list_peudo = [
                    yi + np.random.normal( scale= sigmak_active_ * beta_**(-0.5) ).astype(dtype)[None,:] @ phi_up_list_active[i]
                    for i,yi in enumerate(y_up_list)
                ]
                # update
                yyt_list = [
                    yi@yi.T
                    for i, yi in enumerate(y_up_list_peudo)
                ]
                yyt_list_sum = np.sum(yyt_list)
                ypt_list_active = [
                    yi @ phi_up_list_active[i].T
                    for i,yi in enumerate(y_up_list_peudo)
                ]
                ypt_list_active_sum = np.sum([
                    ypt_list_active[i]
                    for i,yi in enumerate(y_up_list_peudo)
                ], axis=0)            
            else:
                y_up_list_peudo = y_up_list




            w_up_active = mu_vecwup_active.reshape((n_basis_active, n_basis_active)).T
            if (n_iter % n_log == 0) or (n_iter == 1):

                # compute lower bound
                lower_bound = get_lower_bound_active(
                    a_white= a_white,
                    b_white= b_white,
                    a_alpha_active= a_alpha_active,
                    b_alpha_active= b_alpha_active,
                    a_beta_active= a_beta_active,
                    b_beta_active= b_beta_active,

                    a_eta= a_eta,
                    b_eta= b_eta,

                    mu_zi_list_active= mu_zi_list_active,
                    cov_zi_list_active= cov_zi_list_active,
                    mu_vecwup_active= mu_vecwup_active,
                    cov_vecwup_active= cov_vecwup_active,
                    mu_zbar_active= mu_zbar_active,
                    cov_zbar_active= cov_zbar_active,
                    y_up_list= y_up_list,
                    yyt_list_sum= yyt_list_sum,
                    ypt_list_active= ypt_list_active,
                    ppt_list_active= ppt_list_active,

                    a0= self.a0,
                    b0= self.b0,
                )
                self.lower_bound_history_list.append(lower_bound)

                logging.info(f'n_iter={n_iter}, lower_bound={lower_bound:.16e}, sigma={((a_white / b_white)**(-0.5)):.8e}, sigma_k={((a_sigmak_active / b_sigmak_active)**(-0.5))}, \nalpha= {pprint.pformat(a_alpha_active/b_alpha_active, indent=4)}, \nalpha_zbar= {pprint.pformat(a_eta/b_eta, indent=4)}, \nbeta= {pprint.pformat(a_beta_active/b_beta_active, indent=4)}, \nmu_x={mu_zbar_active}, \nw_up={w_up_active}')

                posterior = get_posterior_probability(
                    z_up_list= mu_zi_list_active,
                    w_up= w_up_active,
                    z_up_bar= mu_zbar_active,
                    alpha= a_alpha_active/b_alpha_active,
                    beta= a_beta_active/b_beta_active,
                    eta= a_eta/b_eta,
                    var_white= (a_white/b_white)**2,
                    y_up_list= y_up_list,
                    phi_up_list= phi_up_list_active,
                    a0= self.a0,
                    b0= self.b0,
                )
                self.posterior_history_list.append(posterior)

                if dir_debug is not None:
                    self.a_alpha_active = a_alpha_active
                    self.b_alpha_active = b_alpha_active
                    
                    self.a_beta_active = a_beta_active
                    self.b_beta_active = b_beta_active

                    self.a_eta = a_eta
                    self.b_eta = b_eta

                    self.a_white = a_white
                    self.b_white = b_white
                    
                    self.mu_vecwup_active = mu_vecwup_active
                    self.cov_vecwup_active = cov_vecwup_active

                    self.mu_zi_list_active = mu_zi_list_active
                    self.cov_zi_list_active = cov_zi_list_active

                    self.mu_zbar_active = mu_zbar_active
                    self.cov_zbar_active = cov_zbar_active

                    # parameters specific to the fast fitting
                    self.mu_theta_i_list_active = mu_theta_i_list_active
                    self.cov_theta_i_list_active = cov_theta_i_list_active

                    self.a_sigmak_active = a_sigmak_active
                    self.b_sigmak_active = b_sigmak_active


                    self.k_active_list = k_active_list

                    if param2eval_err_rt is not None:
                        mu_mise, cov_mise = self.plot_results(
                            **param2eval_err_rt,
                            path_fig=f'{dir_debug}/bfpca_predict-n_iter{n_iter}.pdf',
                        )

                        alpha_active_ = self.a_alpha_active/self.b_alpha_active
                        threshold_alpha_ = min( (alpha_active_.min() * self.precision_threshold_multiplier2min), self.max_precision_active_alpha )
                        n_pc_estimate = np.sum( np.logical_not(np.logical_or(
                            alpha_active_ > threshold_alpha_,
                            np.isclose( alpha_active_, threshold_alpha_, )
                        )))

                        iter_time_utime_npc_mucov_error_list.append( ( 
                            n_iter, 
                            time.time() - start_time,
                            resource.getrusage(resource.RUSAGE_SELF).ru_utime - start_resources.ru_utime,
                            n_pc_estimate - param2eval_err_rt['w_up_true'].shape[0],
                            mu_mise, cov_mise 
                            ) )
                        np.savetxt(
                            f'{dir_debug}/iter_time_utime_npc_mucov_error_list.csv',
                            np.array(iter_time_utime_npc_mucov_error_list),
                            delimiter=',',
                            header='iteration, time_in_sec, user_cpu_time_in_sec, diff_n_component, mu_mise, cov_mise',
                            comments='',
                        )                        
                    else:
                        self.plot_results(
                            path_fig=f'{dir_debug}/bfpca_predict-n_iter{n_iter}.pdf',
                        )


            sigma_n2 = a_white / b_white

            
            if update_zbar:
                c = sigma_n2 * ppt_list_active_sum
                np.fill_diagonal(c, c.diagonal() + np.exp( ( np.log(a_eta) - np.log(b_eta) ) + ( np.log(a_beta_active) - np.log(b_beta_active) ) ) )
                cov_zbar_active = svd_inverse(
                    c,
                    cond_max=self.cond_num_max,
                    hermitian= True,
                    logging_prefix = '[cov_zbar_active] ',
                    )

                mu_zbar_active = sigma_n2 * ( ypt_list_active_sum - np.sum([
                    # mu_zi_list_active[i] @ w_up_active @ ppt_list_active[i]
                    mu_theta_i_list_active[i] @ ppt_list_active[i]
                    for i,yi in enumerate(y_up_list_peudo)
                ], axis=0) ) @ cov_zbar_active

                if logging.root.level <= logging.DEBUG:
                    lower_bound = get_lower_bound_active(
                        a_white= a_white,
                        b_white= b_white,
                        a_alpha_active= a_alpha_active,
                        b_alpha_active= b_alpha_active,
                        a_beta_active= a_beta_active,
                        b_beta_active= b_beta_active,

                        a_eta= a_eta,
                        b_eta= b_eta,

                        mu_zi_list_active= mu_zi_list_active,
                        cov_zi_list_active= cov_zi_list_active,
                        mu_vecwup_active= mu_vecwup_active,
                        cov_vecwup_active= cov_vecwup_active,
                        mu_zbar_active= mu_zbar_active,
                        cov_zbar_active= cov_zbar_active,
                        y_up_list= y_up_list,
                        yyt_list_sum= yyt_list_sum,
                        ypt_list_active= ypt_list_active,
                        ppt_list_active= ppt_list_active,
                        
                        a0= self.a0,
                        b0= self.b0,
                    )
                    self.lower_bound_history_list.append(lower_bound)
                    logging.debug(f'n_iter={n_iter}, lower_bound={lower_bound:.16e}, sigma={((a_white / b_white)**(-0.5)):.8e}, sigma_k={((a_sigmak_active / b_sigmak_active)**(-0.5))}, \nalpha= {pprint.pformat(a_alpha_active/b_alpha_active, indent=4)}, \nalpha_zbar= {pprint.pformat(a_eta/b_eta, indent=4)}, \nbeta= {pprint.pformat(a_beta_active/b_beta_active, indent=4)}, \nmu_x={mu_zbar_active}, \nw_up={w_up_active}')

                    posterior = get_posterior_probability(
                        z_up_list= mu_zi_list_active,
                        w_up= w_up_active,
                        z_up_bar= mu_zbar_active,
                        alpha= a_alpha_active/b_alpha_active,
                        beta= a_beta_active/b_beta_active,
                        eta= a_eta/b_eta,
                        var_white= (a_white/b_white)**2,
                        y_up_list= y_up_list,
                        phi_up_list= phi_up_list_active,
                        a0= self.a0,
                        b0= self.b0,
                    )
                    self.posterior_history_list.append(posterior)


            if update_vecwup:

                # update w_up column by column
                sigmak_n2 = np.exp( np.log(a_sigmak_active) - np.log(b_sigmak_active) )
                beta = a_beta_active / b_beta_active

                for k in range(n_basis_active):
                    cov_vecwup_k_active_inv = beta[k] * (
                        sigmak_n2[k] *  np.sum([
                            cov_zi_list_active[i] + mu_zi_list_active[i].T @ mu_zi_list_active[i]
                            for i,yi in enumerate(y_up_list_peudo)
                        ], axis = 0 ) 
                    )
                    np.fill_diagonal(cov_vecwup_k_active_inv, cov_vecwup_k_active_inv.diagonal() + np.exp( np.log(a_alpha_active) - np.log(b_alpha_active) ) )
                    logging.debug(f'inverting cov_vecwup_k_active_inv')
                    cov_vecwup_k_active = svd_inverse(
                        cov_vecwup_k_active_inv,
                        cond_max=self.cond_num_max,
                        hermitian= True,
                        logging_prefix = f'[cov_vecwup_k_active_{k}] ',
                    )
                    cov_vecwup_active[ k*n_basis_active:(k+1)*n_basis_active, k*n_basis_active:(k+1)*n_basis_active ] = cov_vecwup_k_active
                    
                    mu_vecwup_k_active = sigmak_n2[k] * beta[k] * np.sum([
                        mu_theta_i_list_active[i][0,k] * mu_zi_list_active[i]
                        for i,yi in enumerate(y_up_list_peudo)
                    ], axis=0) @ cov_vecwup_k_active
                    mu_vecwup_active[ :, k*n_basis_active:(k+1)*n_basis_active ] = mu_vecwup_k_active

                '''
                # weight matrix
                cov_vecwup_active_inv = (
                    sigma_n2 * np.sum([
                            np.kron( ppt_list_active[i], cov_zi_list_active[i] + mu_zi_list_active[i].T @ mu_zi_list_active[i] )
                            for i,yi in enumerate(y_up_list)
                        ], axis=0) + \
                            np.diag( np.exp( ( (np.log(a_beta_active) - np.log(b_beta_active))[None,:].T + (np.log(a_alpha_active) - np.log(b_alpha_active))[None,:] ).reshape(-1) ) )
                )
                logging.debug(f'inverting cov_vecwup_active_inv')
                cov_vecwup_active = svd_inverse(
                    cov_vecwup_active_inv,
                    cond_max=self.cond_num_max,
                    hermitian= True,
                    logging_prefix = '[cov_vecwup_active] ',
                )
                mu_vecwup_active = sigma_n2 * (np.sum([
                    ( mu_zi_list_active[i].T @ ( ypt_list_active[i] - mu_zbar_active @ ppt_list_active[i] ) ).T.reshape((1,-1))
                    for i,yi in enumerate(y_up_list)
                ], axis=0)) @ cov_vecwup_active
                '''
                
                if logging.root.level <= logging.DEBUG:
                    lower_bound = get_lower_bound_active(
                        a_white= a_white,
                        b_white= b_white,
                        a_alpha_active= a_alpha_active,
                        b_alpha_active= b_alpha_active,
                        a_beta_active= a_beta_active,
                        b_beta_active= b_beta_active,

                        a_eta= a_eta,
                        b_eta= b_eta,

                        mu_zi_list_active= mu_zi_list_active,
                        cov_zi_list_active= cov_zi_list_active,
                        mu_vecwup_active= mu_vecwup_active,
                        cov_vecwup_active= cov_vecwup_active,
                        mu_zbar_active= mu_zbar_active,
                        cov_zbar_active= cov_zbar_active,
                        y_up_list= y_up_list,
                        yyt_list_sum= yyt_list_sum,
                        ypt_list_active= ypt_list_active,
                        ppt_list_active= ppt_list_active,
                        
                        a0= self.a0,
                        b0= self.b0,
                    )
                    self.lower_bound_history_list.append(lower_bound)
                    w_up_active = mu_vecwup_active.reshape((n_basis_active, n_basis_active)).T
                    logging.debug(f'n_iter={n_iter}, lower_bound={lower_bound:.16e}, sigma={((a_white / b_white)**(-0.5)):.8e}, sigma_k={((a_sigmak_active / b_sigmak_active)**(-0.5))}, \nalpha= {pprint.pformat(a_alpha_active/b_alpha_active, indent=4)}, \nalpha_zbar= {pprint.pformat(a_eta/b_eta, indent=4)}, \nbeta= {pprint.pformat(a_beta_active/b_beta_active, indent=4)}, \nmu_x={mu_zbar_active}, \nw_up={w_up_active}')

                    posterior = get_posterior_probability(
                        z_up_list= mu_zi_list_active,
                        w_up= w_up_active,
                        z_up_bar= mu_zbar_active,
                        alpha= a_alpha_active/b_alpha_active,
                        beta= a_beta_active/b_beta_active,
                        eta= a_eta/b_eta,
                        var_white= (a_white/b_white)**2,
                        y_up_list= y_up_list,
                        phi_up_list= phi_up_list_active,
                        a0= self.a0,
                        b0= self.b0,
                    )
                    self.posterior_history_list.append(posterior)


            w_up_active = mu_vecwup_active.reshape((n_basis_active, n_basis_active)).T


            # NOTE n_basis_inactive is the number of basis functions that are included but not active 
            b_alpha_active_old = b_alpha_active.copy()
            alpha_change_small = False
            if update_alpha:
                # alpha
                for j in range(n_basis_active):
                    m = w_up_active[ (j,), : ]
                    b_alpha_active_j_old = b_alpha_active[j].copy()
                    b_alpha_active[j] = 1/2 * ( ( ( (m**2).flatten() + np.diag( cov_vecwup_active[ j::n_basis_active, j::n_basis_active] ) ) * ( a_beta_active / b_beta_active ) ).sum() ) + self.b0
                    
                    log10_gap_min_alpha_beta = abs( np.log10(np.min( a_alpha_active / b_alpha_active )) - np.log10(np.min( a_beta_active / b_beta_active )) )
                    if log10_gap_min_alpha_beta > max_log10_gap_min_alpha_beta:
                        # revert the update
                        b_alpha_active[j] = b_alpha_active_j_old
                
                # NOTE this is just for convenience. it usually optimizes the objective function but is not gurateed. the same applyies to udpates for beta, eta
                # threshold large precisions to prevent numerical issues
                alpha_active_expect = a_alpha_active/b_alpha_active
                if_alpha_is_bounded = ( alpha_active_expect >= self.max_precision_active_alpha )
                b_alpha_active[ if_alpha_is_bounded ] = ( a_alpha_active / self.max_precision_active_alpha )[ if_alpha_is_bounded ]

                # # force some alpha to be inactive in early optimization

                rc = np.abs(b_alpha_active_old - b_alpha_active) / b_alpha_active_old
                if (n_iter % n_log == 0) or (n_iter == 1):
                    logging.debug(f'np.abs(b_alpha_active_old - b_alpha_active) / b_alpha_active_old={rc}')
                if ( (np.max(rc) < tol_converge_rel_alpha_beta) and ( n_iter > max( n_iter_max_delaywhite, sigmak_fix_iter, n_iter_anneal ) ) ):
                    alpha_change_small = True

                if logging.root.level <= logging.DEBUG:
                    lower_bound = get_lower_bound_active(
                        a_white= a_white,
                        b_white= b_white,
                        a_alpha_active= a_alpha_active,
                        b_alpha_active= b_alpha_active,
                        a_beta_active= a_beta_active,
                        b_beta_active= b_beta_active,

                        a_eta= a_eta,
                        b_eta= b_eta,

                        mu_zi_list_active= mu_zi_list_active,
                        cov_zi_list_active= cov_zi_list_active,
                        mu_vecwup_active= mu_vecwup_active,
                        cov_vecwup_active= cov_vecwup_active,
                        mu_zbar_active= mu_zbar_active,
                        cov_zbar_active= cov_zbar_active,
                        y_up_list= y_up_list,
                        yyt_list_sum= yyt_list_sum,
                        ypt_list_active= ypt_list_active,
                        ppt_list_active= ppt_list_active,
                        
                        a0= self.a0,
                        b0= self.b0,
                    )
                    self.lower_bound_history_list.append(lower_bound)
                    logging.debug(f'n_iter={n_iter}, lower_bound={lower_bound:.16e}, sigma={((a_white / b_white)**(-0.5)):.8e}, sigma_k={((a_sigmak_active / b_sigmak_active)**(-0.5))}, \nalpha= {pprint.pformat(a_alpha_active/b_alpha_active, indent=4)}, \nalpha_zbar= {pprint.pformat(a_eta/b_eta, indent=4)}, \nbeta= {pprint.pformat(a_beta_active/b_beta_active, indent=4)}, \nmu_x={mu_zbar_active}, \nw_up={w_up_active}')


                    posterior = get_posterior_probability(
                        z_up_list= mu_zi_list_active,
                        w_up= w_up_active,
                        z_up_bar= mu_zbar_active,
                        alpha= a_alpha_active/b_alpha_active,
                        beta= a_beta_active/b_beta_active,
                        eta= a_eta/b_eta,
                        var_white= (a_white/b_white)**2,
                        y_up_list= y_up_list,
                        phi_up_list= phi_up_list_active,
                        a0= self.a0,
                        b0= self.b0,
                    )
                    self.posterior_history_list.append(posterior)
            
            alpha_active_relachange.append(
                np.log10(np.max([
                    np.abs( b_alpha_active / b_alpha_active_old - 1 )
                ]) + 1e-10)
            )

            if update_eta:
                b_eta = 1/2 * ( ( (mu_zbar_active**2).flatten() + np.diag(cov_zbar_active) ) * ( a_beta_active / b_beta_active ) ).sum() + self.b0

                if (a_eta/b_eta > self.max_precision_active_alpha):
                    b_eta = ( a_eta / self.max_precision_active_alpha ) 
                # threshold large precisions to prevent numerical issues

                if logging.root.level <= logging.DEBUG:
                    lower_bound = get_lower_bound_active(
                        a_white= a_white,
                        b_white= b_white,
                        a_alpha_active= a_alpha_active,
                        b_alpha_active= b_alpha_active,
                        a_beta_active= a_beta_active,
                        b_beta_active= b_beta_active,

                        a_eta= a_eta,
                        b_eta= b_eta,

                        mu_zi_list_active= mu_zi_list_active,
                        cov_zi_list_active= cov_zi_list_active,
                        mu_vecwup_active= mu_vecwup_active,
                        cov_vecwup_active= cov_vecwup_active,
                        mu_zbar_active= mu_zbar_active,
                        cov_zbar_active= cov_zbar_active,
                        y_up_list= y_up_list,
                        yyt_list_sum= yyt_list_sum,
                        ypt_list_active= ypt_list_active,
                        ppt_list_active= ppt_list_active,
                        
                        a0= self.a0,
                        b0= self.b0,
                    )
                    self.lower_bound_history_list.append(lower_bound)
                    logging.debug(f'n_iter={n_iter}, lower_bound={lower_bound:.16e}, sigma={((a_white / b_white)**(-0.5)):.8e}, sigma_k={((a_sigmak_active / b_sigmak_active)**(-0.5))}, \nalpha= {pprint.pformat(a_alpha_active/b_alpha_active, indent=4)}, \nalpha_zbar= {pprint.pformat(a_eta/b_eta, indent=4)}, \nbeta= {pprint.pformat(a_beta_active/b_beta_active, indent=4)}, \nmu_x={mu_zbar_active}, \nw_up={w_up_active}')


                    posterior = get_posterior_probability(
                        z_up_list= mu_zi_list_active,
                        w_up= w_up_active,
                        z_up_bar= mu_zbar_active,
                        alpha= a_alpha_active/b_alpha_active,
                        beta= a_beta_active/b_beta_active,
                        eta= a_eta/b_eta,
                        var_white= (a_white/b_white)**2,
                        y_up_list= y_up_list,
                        phi_up_list= phi_up_list_active,
                        a0= self.a0,
                        b0= self.b0,
                    )
                    self.posterior_history_list.append(posterior)

            if update_z_up:

                '''
                ci_list = []
                for i,yi in enumerate(y_up_list):
                    c = np.zeros(shape=(n_basis_active, n_basis_active))
                    for j in range(n_basis_active):
                        for k in range(n_basis_active):
                            c[j,k] = float( vectorize_matrix( w_up_active[ [k,], : ].T @ w_up_active[ [j,], : ] + cov_vecwup_active[ j::n_basis_active, k::n_basis_active ] ).T @ vectorize_matrix( ppt_list_active[i] ) )

                    ci_list.append(c)

                # latent variables
                for i,yi in enumerate(y_up_list):
                    
                    cov_zi_list_active[i] = svd_inverse(
                        ( sigma_n2 * ci_list[i] + np.identity(n= n_basis_active) ),
                        cond_max=self.cond_num_max,
                        hermitian= True,
                        logging_prefix = f'[cov_zi_list_active[{i}]] ',
                        )
                    mu_zi_list_active[i] = sigma_n2 * ( ( ypt_list_active[i] - mu_zbar_active @ ppt_list_active[i] ) @ w_up_active.T ) @ cov_zi_list_active[i]
                '''

                w_up_active = mu_vecwup_active.reshape((n_basis_active, n_basis_active)).T
                sigmak_n2 = np.exp( ( np.log(a_sigmak_active) - np.log(b_sigmak_active) ) )
                beta = a_beta_active / b_beta_active

                cov_zi_list_active_ = svd_inverse(
                    w_up_active * ( sigmak_n2 * beta )[None,:] @ w_up_active.T + np.sum([
                        sigmak_n2[k] * beta[k] * cov_vecwup_active[ k*n_basis_active:(k+1)*n_basis_active, k*n_basis_active:(k+1)*n_basis_active ]
                        for k in range(n_basis_active)
                    ], axis=0) + np.identity(n= n_basis_active, dtype=dtype),
                    hermitian= True,
                )
                for i,yi in enumerate(y_up_list_peudo):
                    cov_zi_list_active[i] = cov_zi_list_active_.copy()

                    mu_zi_list_active[i] = mu_theta_i_list_active[i] * ( sigmak_n2 * beta )[None,:] @ w_up_active.T @ cov_zi_list_active[i]


                if logging.root.level <= logging.DEBUG:
                    lower_bound = get_lower_bound_active(
                        a_white= a_white,
                        b_white= b_white,
                        a_alpha_active= a_alpha_active,
                        b_alpha_active= b_alpha_active,
                        a_beta_active= a_beta_active,
                        b_beta_active= b_beta_active,

                        a_eta= a_eta,
                        b_eta= b_eta,

                        mu_zi_list_active= mu_zi_list_active,
                        cov_zi_list_active= cov_zi_list_active,
                        mu_vecwup_active= mu_vecwup_active,
                        cov_vecwup_active= cov_vecwup_active,
                        mu_zbar_active= mu_zbar_active,
                        cov_zbar_active= cov_zbar_active,
                        y_up_list= y_up_list,
                        yyt_list_sum= yyt_list_sum,
                        ypt_list_active= ypt_list_active,
                        ppt_list_active= ppt_list_active,
                        
                        a0= self.a0,
                        b0= self.b0,
                    )
                    self.lower_bound_history_list.append(lower_bound)
                    logging.debug(f'n_iter={n_iter}, lower_bound={lower_bound:.16e}, sigma={((a_white / b_white)**(-0.5)):.8e}, sigma_k={((a_sigmak_active / b_sigmak_active)**(-0.5))}, \nalpha= {pprint.pformat(a_alpha_active/b_alpha_active, indent=4)}, \nalpha_zbar= {pprint.pformat(a_eta/b_eta, indent=4)}, \nbeta= {pprint.pformat(a_beta_active/b_beta_active, indent=4)}, \nmu_x={mu_zbar_active}, \nw_up={w_up_active}')

                    posterior = get_posterior_probability(
                        z_up_list= mu_zi_list_active,
                        w_up= w_up_active,
                        z_up_bar= mu_zbar_active,
                        alpha= a_alpha_active/b_alpha_active,
                        beta= a_beta_active/b_beta_active,
                        eta= a_eta/b_eta,
                        var_white= (a_white/b_white)**2,
                        y_up_list= y_up_list,
                        phi_up_list= phi_up_list_active,
                        a0= self.a0,
                        b0= self.b0,
                    )
                    self.posterior_history_list.append(posterior)

            mu_theta_i_list_active_old = [e.copy() for e in mu_theta_i_list_active]
            if update_theta:
                
                sigmak_n2 = np.exp( ( np.log(a_sigmak_active) - np.log(b_sigmak_active) ) )
                beta = a_beta_active / b_beta_active
                sigma_n2 = a_white / b_white

                cov_theta_i_list_active_pinv = [None for i in range(len(cov_theta_i_list_active))]
                for i,yi in enumerate(y_up_list_peudo):
                    p = ppt_list_active[i] * ( a_white / b_white )
                    np.fill_diagonal(p, p.diagonal() + sigmak_n2 * beta)
                    cov_theta_i_list_active[i], cov_theta_i_list_active_pinv[i] = svd_inverse( 
                        p, cond_max=self.cond_num_max, hermitian= True,
                        rcond_pinv = self.rcond_pinv,
                        )
                    mu_theta_i_list_active[i] = (
                        ( y_up_list_peudo[i] @ phi_up_list_active[i].T - mu_zbar_active @ ppt_list_active[i] ) * sigma_n2 + mu_zi_list_active[i] @ w_up_active * ( sigmak_n2 * beta )[None,:]
                    ) @ cov_theta_i_list_active_pinv[i]
            mu_theta_i_list_active_relachange.append(
                np.log10(np.max([
                    np.abs( (mu_theta_i_list_active[i] - mu_theta_i_list_active_old[i]) / mu_theta_i_list_active_old[i] )
                    for i,yi in enumerate(y_up_list_peudo)
                ]) + 1e-10)
            )
            theta_minus_zw_norm_list.append(
                np.sum([
                    np.linalg.norm( mu_theta_i_list_active[i] - mu_zi_list_active[i] @ w_up_active )
                    for i,yi in enumerate(y_up_list_peudo)
                ])
            )

            
            b_beta_active_old = b_beta_active.copy()
            if update_beta:
    
                # beta
                sigmak_n2 = np.exp( ( np.log(a_sigmak_active) - np.log(b_sigmak_active) ) )
                for k in range(n_basis_active):
                    m = w_up_active[ :, (k,) ]
                    b_beta_active_k_old = b_beta_active[k].copy()
                    
                    b_beta_active[k] = 1/2 * ( ( ( (m**2).flatten() + np.diag( cov_vecwup_active[ k*n_basis_active: (k+1)*n_basis_active, k*n_basis_active: (k+1)*n_basis_active ] ) ) * ( a_alpha_active / b_alpha_active ) ).sum() ) + self.b0

                    b_beta_active[k] = b_beta_active[k] + ( a_eta / b_eta ) * 0.5 * (mu_zbar_active[0,k]**2 + cov_zbar_active[k,k])

                    # additional term for the fast fitting
                    # b_beta_active[k] = b_beta_active[k] + 0.5 * sigmak_n2[k] * (
                    #     np.sum([
                    #         ( mu_theta_i_list_active[i][0,k]**2 + cov_theta_i_list_active[i][k,k] ) \
                    #         + np.trace( (
                    #             cov_zi_list_active[i] + mu_zi_list_active[i].T @ mu_zi_list_active[i]
                    #         ) @ (
                    #             cov_vecwup_active[ k*n_basis_active: (k+1)*n_basis_active, k*n_basis_active: (k+1)*n_basis_active ] + m @ m.T
                    #         ) ) - 2 * mu_theta_i_list_active[i][0,k] * mu_zi_list_active[i] @ m
                    #         for i,yi in enumerate(y_up_list_peudo)
                    #     ])
                    # )
                    # adjust the formular to prevent numerical issues, guaranteeing the positive definiteness
                    b_beta_active[k] = b_beta_active[k] + 0.5 * sigmak_n2[k] * np.sum([
                            ( mu_theta_i_list_active[i][0,k] - mu_zi_list_active[i] @ m )**2 + cov_theta_i_list_active[i][k,k] + np.trace(
                                ( cov_zi_list_active[i] + mu_zi_list_active[i].T @ mu_zi_list_active[i] ) @ cov_vecwup_active[ k*n_basis_active: (k+1)*n_basis_active, k*n_basis_active: (k+1)*n_basis_active ] + cov_zi_list_active[i] @ m @ m.T
                            )
                            for i,yi in enumerate(y_up_list_peudo)
                        ])
                    log10_gap_min_alpha_beta = abs( np.log10(np.min( a_alpha_active / b_alpha_active )) - np.log10(np.min( a_beta_active / b_beta_active )) )
                    if log10_gap_min_alpha_beta > max_log10_gap_min_alpha_beta:

                        # revert the update
                        b_beta_active[k] = b_beta_active_k_old
                # threshold large precisions to prevent numerical issues
                beta_active_expect = a_beta_active/b_beta_active
                if_beta_is_bounded = ( beta_active_expect >= self.max_precision_active_beta )
                b_beta_active[ if_beta_is_bounded ] = ( a_beta_active / self.max_precision_active_beta )[ if_beta_is_bounded ]

                logging.debug(f'n_iter={n_iter}, beta, \nbeta= {pprint.pformat(beta_active_expect, indent=4)}')

                if logging.root.level <= logging.DEBUG:
                    lower_bound = get_lower_bound_active(
                        a_white= a_white,
                        b_white= b_white,
                        a_alpha_active= a_alpha_active,
                        b_alpha_active= b_alpha_active,
                        a_beta_active= a_beta_active,
                        b_beta_active= b_beta_active,

                        a_eta= a_eta,
                        b_eta= b_eta,

                        mu_zi_list_active= mu_zi_list_active,
                        cov_zi_list_active= cov_zi_list_active,
                        mu_vecwup_active= mu_vecwup_active,
                        cov_vecwup_active= cov_vecwup_active,
                        mu_zbar_active= mu_zbar_active,
                        cov_zbar_active= cov_zbar_active,
                        y_up_list= y_up_list,
                        yyt_list_sum= yyt_list_sum,
                        ypt_list_active= ypt_list_active,
                        ppt_list_active= ppt_list_active,
                        
                        a0= self.a0,
                        b0= self.b0,
                    )
                    self.lower_bound_history_list.append(lower_bound)
                    logging.debug(f'n_iter={n_iter}, lower_bound={lower_bound:.16e}, sigma={((a_white / b_white)**(-0.5)):.8e}, sigma_k={((a_sigmak_active / b_sigmak_active)**(-0.5))}, \nalpha= {pprint.pformat(a_alpha_active/b_alpha_active, indent=4)}, \nalpha_zbar= {pprint.pformat(a_eta/b_eta, indent=4)}, \nbeta= {pprint.pformat(a_beta_active/b_beta_active, indent=4)}, \nmu_x={mu_zbar_active}, \nw_up={w_up_active}')

                    posterior = get_posterior_probability(
                        z_up_list= mu_zi_list_active,
                        w_up= w_up_active,
                        z_up_bar= mu_zbar_active,
                        alpha= a_alpha_active/b_alpha_active,
                        beta= a_beta_active/b_beta_active,
                        eta= a_eta/b_eta,
                        var_white= (a_white/b_white)**2,
                        y_up_list= y_up_list,
                        phi_up_list= phi_up_list_active,
                        a0= self.a0,
                        b0= self.b0,
                    )
                    self.posterior_history_list.append(posterior)

            rc = np.max([
                np.abs( b_beta_active / b_beta_active_old - 1 )
            ])
            beta_active_relachange.append(np.log10(rc  + 1e-10 ))
            if alpha_change_small and ( rc < tol_converge_rel_alpha_beta ):
                n_iter_converge_plateau_cntdw -= 1
            else:
                n_iter_converge_plateau_cntdw = n_iter_converge_plateau

            update_sigmak = False
            if update_sigmak:
                # TODO, HACK, sigmak is not updated
                pass            

            if update_white and \
                (n_iter >= n_iter_max_delaywhite_):

                '''                
                d_up = np.zeros(shape=(n_basis_active**2, n_basis_active**2))
                for j in range(n_basis_active):
                    for k in range(n_basis_active):
                        d_up[ (j+k*n_basis_active, ), : ] = vectorize_matrix(( w_up_active[ [k,],  : ].T @ w_up_active[ [j,],  : ] + cov_vecwup_active[ j::n_basis_active, k::n_basis_active ] ).T).T

                b_white = 0.5 * yyt_list_sum - w_up_active.T.reshape(1,-1) @ np.sum([
                    mu_zi_list_active[i].T @ ypt_list_active[i]
                    for i,yi in enumerate(y_up_list)
                ], axis=0).T.reshape(1,-1).T + 0.5 * vectorize_matrix( d_up.T ).T @ vectorize_matrix(np.sum([
                    vectorize_matrix( ppt_list_active[i] ) @ vectorize_matrix( cov_zi_list_active[i] + mu_zi_list_active[i].T @ mu_zi_list_active[i] ).T
                    for i,yi in enumerate(y_up_list)
                ], axis=0)) - ( ypt_list_active_sum - np.sum([
                    ( mu_zi_list_active[i] @ w_up_active @ ppt_list_active[i] )
                    for i,yi in enumerate(y_up_list)
                ], axis=0) ) @ mu_zbar_active.T  + 0.5 * ( cov_zbar_active + mu_zbar_active.T @ mu_zbar_active ).T.reshape(1,-1) @ ppt_list_active_sum.T.reshape(1,-1).T
                b_white = float(b_white) + self.b0
                '''

                b_white = 0.5 * yyt_list_sum + 0.5 * np.sum([
                    np.trace(
                        ((
                            mu_zbar_active.T @ mu_zbar_active + cov_zbar_active
                        ) + (
                            mu_theta_i_list_active[i].T @ mu_theta_i_list_active[i] + cov_theta_i_list_active[i]
                        )) @ ppt_list_active[i]
                    ) - 2 * ( mu_zbar_active + mu_theta_i_list_active[i] ) @ ypt_list_active[i].T \
                    + 2 * ( mu_zbar_active @ ppt_list_active[i] @ mu_theta_i_list_active[i].T )
                    for i,yi in enumerate(y_up_list_peudo)
                    ], axis=0) 
                b_white = float(b_white) + self.b0

                if logging.root.level <= logging.DEBUG:
                    lower_bound = get_lower_bound_active(
                        a_white= a_white,
                        b_white= b_white,
                        a_alpha_active= a_alpha_active,
                        b_alpha_active= b_alpha_active,
                        a_beta_active= a_beta_active,
                        b_beta_active= b_beta_active,

                        a_eta= a_eta,
                        b_eta= b_eta,

                        mu_zi_list_active= mu_zi_list_active,
                        cov_zi_list_active= cov_zi_list_active,
                        mu_vecwup_active= mu_vecwup_active,
                        cov_vecwup_active= cov_vecwup_active,
                        mu_zbar_active= mu_zbar_active,
                        cov_zbar_active= cov_zbar_active,
                        y_up_list= y_up_list,
                        yyt_list_sum= yyt_list_sum,
                        ypt_list_active= ypt_list_active,
                        ppt_list_active= ppt_list_active,
                        
                        a0= self.a0,
                        b0= self.b0,
                    )
                    self.lower_bound_history_list.append(lower_bound)
                    logging.debug(f'n_iter={n_iter}, lower_bound={lower_bound:.16e}, sigma={((a_white / b_white)**(-0.5)):.8e}, sigma_k={((a_sigmak_active / b_sigmak_active)**(-0.5))}, \nalpha= {pprint.pformat(a_alpha_active/b_alpha_active, indent=4)}, \nalpha_zbar= {pprint.pformat(a_eta/b_eta, indent=4)}, \nbeta= {pprint.pformat(a_beta_active/b_beta_active, indent=4)}, \nmu_x={mu_zbar_active}, \nw_up={w_up_active}')

                    posterior = get_posterior_probability(
                        z_up_list= mu_zi_list_active,
                        w_up= w_up_active,
                        z_up_bar= mu_zbar_active,
                        alpha= a_alpha_active/b_alpha_active,
                        beta= a_beta_active/b_beta_active,
                        eta= a_eta/b_eta,
                        var_white= (a_white/b_white)**2,
                        y_up_list= y_up_list,
                        phi_up_list= phi_up_list_active,
                        a0= self.a0,
                        b0= self.b0,
                    )
                    self.posterior_history_list.append(posterior)




            # check for convergence
            lower_bound = get_lower_bound_active(
                a_white= a_white,
                b_white= b_white,
                a_alpha_active= a_alpha_active,
                b_alpha_active= b_alpha_active,
                a_beta_active= a_beta_active,
                b_beta_active= b_beta_active,

                a_eta= a_eta,
                b_eta= b_eta,

                mu_zi_list_active= mu_zi_list_active,
                cov_zi_list_active= cov_zi_list_active,
                mu_vecwup_active= mu_vecwup_active,
                cov_vecwup_active= cov_vecwup_active,
                mu_zbar_active= mu_zbar_active,
                cov_zbar_active= cov_zbar_active,
                y_up_list= y_up_list,
                yyt_list_sum= yyt_list_sum,
                ypt_list_active= ypt_list_active,
                ppt_list_active= ppt_list_active,
                
                a0= self.a0,
                b0= self.b0,
            )
            self.lower_bound_history_list.append(lower_bound)

            posterior = get_posterior_probability(
                z_up_list= mu_zi_list_active,
                w_up= w_up_active,
                z_up_bar= mu_zbar_active,
                alpha= a_alpha_active/b_alpha_active,
                beta= a_beta_active/b_beta_active,
                eta= a_eta/b_eta,
                var_white= (a_white/b_white)**2,
                y_up_list= y_up_list,
                phi_up_list= phi_up_list_active,
                a0= self.a0,
                b0= self.b0,
            )
            self.posterior_history_list.append(posterior)

            # sanity check, lower bound should never decrease
            lbhl = np.array(self.lower_bound_history_list)
            if not np.all(lbhl[1:] >= lbhl[:-1]):
                d = lbhl[1:] - lbhl[:-1]
                decrease_n_iter = [
                    e for e in list(zip( d[d<0], np.arange(len(lbhl[1:]))[d<0]+1, )) 
                    if ( e[1] not in ( self.n_lbhist_new_basis_list + self.n_lbhist_remove_basis_list ) )
                ]
                if len(decrease_n_iter) > 0:
                    logging.warning(f'lower bound decreased!\n{sorted(decrease_n_iter)}')
            # TODO, HACK, this currently does not try new basis functions
            # n_iter_max_delaywhite_ = 0
            if n_iter_converge_plateau_cntdw <= 0:
            # if (lower_bound-lower_bound_old) < tol_convergence:
                # logging.info(f'lower_bound-lower_bound_old = {lower_bound-lower_bound_old:.3e} < tol_convergence = {tol_convergence:.3e}')

                # if n_iter>n_iter_max_delaywhite_:

                    # if n_patience_countdown <= 0:
                    #     logging.info(f'n_patience_countdown={n_patience_countdown} is used up')

                # try more basis functions
                logging.info(f'try more basis functions')
                continue_run = False
                if (sim_threshold_new_basis > -1):
                    self_try = self.try_activate_new_beta(
                        mu_zi_list_active = mu_zi_list_active.copy(),
                        cov_zi_list_active= cov_zi_list_active,
                        mu_vecwup_active= mu_vecwup_active.copy(),
                        cov_vecwup_active= cov_vecwup_active.copy(),
                        mu_zbar_active= mu_zbar_active.copy(),
                        cov_zbar_active= cov_zbar_active.copy(),
                        alpha_active= (a_alpha_active/b_alpha_active).copy(),
                        beta_active= (a_beta_active / b_beta_active).copy(),
                        sigma_white= (a_white / b_white)**(-0.5),

                        eta= a_eta/b_eta,

                        mu_theta_i_list_active= mu_theta_i_list_active.copy(),
                        cov_theta_i_list_active= cov_theta_i_list_active.copy(),

                        sigmak_active= (a_sigmak_active / b_sigmak_active)**(-0.5),

                        phi_up_list_active = phi_up_list_active,

                        update_white = update_white,
                        update_vecwup = update_vecwup,
                        update_z_up = update_z_up,
                        update_zbar = update_zbar,
                        update_alpha = update_alpha,
                        update_eta = update_eta,
                        update_theta = update_theta,

                        update_beta= update_beta,

                        sim_threshold = sim_threshold_new_basis,

                        tol_converge_rel_alpha_beta = tol_converge_rel_alpha_beta,
                        n_iter_converge_plateau = n_iter_converge_plateau,

                        use_pseudo_y = use_pseudo_y,
                        new_basis_corr_dataset_size= new_basis_corr_dataset_size,

                        n_iter_max= 3,
                        n_iter_max_delaywhite= 0,
                        n_log= 1,

                        tol_convergence= tol_convergence,
                        dir_debug= f'{dir_debug}/addbase/n_iter{n_iter}' if dir_debug is not None else None,
                    )

                    if (self_try is not None):

                        continue_run = True
                        n_iter_converge_plateau_cntdw = n_iter_converge_plateau

                        k_active_list= self_try.k_active_list
                        n_basis_active = len( k_active_list )
                        self.k_active_list = k_active_list
                        a_white= self_try.a_white
                        b_white= self_try.b_white
                        a_alpha_active= self_try.a_alpha_active
                        b_alpha_active= self_try.b_alpha_active
                        a_eta= self_try.a_eta
                        b_eta= self_try.b_eta
                        a_beta_active= self_try.a_beta_active
                        b_beta_active= self_try.b_beta_active
                        a_sigmak_active= self_try.a_sigmak_active
                        b_sigmak_active= self_try.b_sigmak_active
                        mu_zi_list_active= self_try.mu_zi_list_active
                        cov_zi_list_active= self_try.cov_zi_list_active
                        mu_vecwup_active= self_try.mu_vecwup_active
                        cov_vecwup_active= self_try.cov_vecwup_active
                        w_up_active = mu_vecwup_active.reshape((n_basis_active, n_basis_active)).T
                        mu_zbar_active= self_try.mu_zbar_active
                        cov_zbar_active= self_try.cov_zbar_active
                        mu_theta_i_list_active= self_try.mu_theta_i_list_active
                        cov_theta_i_list_active= self_try.cov_theta_i_list_active
                        self.n_lbhist_new_basis_list.append( len(self.lower_bound_history_list) )
                        self.lower_bound_history_list += self_try.lower_bound_history_list
                        self.posterior_history_list += self_try.posterior_history_list

                        # update the precomputed
                        phi_up_list_active = [
                            np.concatenate([
                                self.kernel_list[ k // n_index_basis ]( self.x_base_index[ [k % n_index_basis] ], zi[0] )
                                for k in self.k_active_list
                            ], axis =0)
                            for i, zi in enumerate(self.x_up_list)
                        ]
                        ppt_list_active = [
                            pi @ pi.T
                            for i,pi in enumerate(phi_up_list_active)
                        ]
                        ppt_list_active_sum = (np.sum(ppt_list_active, axis=0))
                        # ppt_list_active_sum_inv = svd_inverse(
                        #     ppt_list_active_sum, 
                        #     cond_max=self.cond_num_max, 
                        #     hermitian= True,
                        #     logging_prefix = f'[ppt_list_active_sum_inv] ',
                        #     )
                        ypt_list_active = [
                            yi @ phi_up_list_active[i].T
                            for i,yi in enumerate(y_up_list_peudo)
                        ]
                        ypt_list_active_sum = np.sum([
                            ypt_list_active[i]
                            for i,yi in enumerate(y_up_list_peudo)
                        ], axis=0)

                        logging.info(f'continue optimization, k_active_list={self_try.k_active_list}')


                if not continue_run:
                    converged = True
                    logging.info(f'Converged!')

                    # else:
                    #     n_patience_countdown -= 1
                    #     logging.info(f'n_patience_countdown={n_patience_countdown}')

                # else:
                #     n_iter_max_delaywhite_ = 0
                #     logging.info(f'Converged with all but sigma for white noise! Now starting optimization with white noise sigma')
            
            # else:
            #     # reset n_patience_countdown
            #     n_patience_countdown = n_patience


            # lower bound might be changed by the actived alpha
            lower_bound = get_lower_bound_active(
                a_white= a_white,
                b_white= b_white,
                a_alpha_active= a_alpha_active,
                b_alpha_active= b_alpha_active,
                a_beta_active= a_beta_active,
                b_beta_active= b_beta_active,

                a_eta= a_eta,
                b_eta= b_eta,

                mu_zi_list_active= mu_zi_list_active,
                cov_zi_list_active= cov_zi_list_active,
                mu_vecwup_active= mu_vecwup_active,
                cov_vecwup_active= cov_vecwup_active,
                mu_zbar_active= mu_zbar_active,
                cov_zbar_active= cov_zbar_active,
                y_up_list= y_up_list,
                yyt_list_sum= yyt_list_sum,
                ypt_list_active= ypt_list_active,
                ppt_list_active= ppt_list_active,
                
                a0= self.a0,
                b0= self.b0,
            )
            self.lower_bound_history_list.append(lower_bound)

            w_up_active = mu_vecwup_active.reshape((n_basis_active, n_basis_active)).T
            posterior = get_posterior_probability(
                z_up_list= mu_zi_list_active,
                w_up= w_up_active,
                z_up_bar= mu_zbar_active,
                alpha= a_alpha_active/b_alpha_active,
                beta= a_beta_active/b_beta_active,
                eta= a_eta/b_eta,
                var_white= (a_white/b_white)**2,
                y_up_list= y_up_list,
                phi_up_list= phi_up_list_active,
                a0= self.a0,
                b0= self.b0,
            )
            self.posterior_history_list.append(posterior)


            # remove the active basis functions with bounded precision
            # HACK, TODO, this currently does not remove the basis functions
            if remove_redundant_basis:
                if len(k_active_list) > 1:

                    idx_beta_less = (np.where( np.logical_not(np.logical_or(
                        a_beta_active/b_beta_active > self.max_precision_active_beta,
                        np.isclose( a_beta_active/b_beta_active, self.max_precision_active_beta ),
                    )) )[0].tolist())

                    idx_alpha_less = (np.where( np.logical_not(np.logical_or(
                        a_alpha_active/b_alpha_active > self.max_precision_active_alpha,
                        np.isclose( a_alpha_active/b_alpha_active, self.max_precision_active_alpha ),
                    )) )[0].tolist())
                    n_remain = max( len(idx_beta_less), len(idx_alpha_less) )

                    logging.debug(f'idx_alpha_less={idx_alpha_less}')
                    logging.debug(f'idx_beta_less={idx_beta_less}')

                    if (0 < n_remain) and (n_remain < len(k_active_list)):

                        logging.info(f'removing redundant basis functions')

                        idx_alpha_remain =  (np.argsort( a_alpha_active/b_alpha_active )[:n_remain].tolist())
                        idx_alpha_remain = sorted(idx_alpha_remain)
                        logging.debug(f'idx_alpha_remain={idx_alpha_remain}')
                        idx_beta_remain =  (np.argsort( a_beta_active/b_beta_active )[:n_remain].tolist())
                        idx_beta_remain = sorted(idx_beta_remain)
                        logging.debug(f'idx_beta_remain={idx_beta_remain}')


                        idx_vecwup_remain = np.array([
                            [
                                (ia + ib * len(k_active_list)) for ia in idx_alpha_remain
                            ]
                            for ib in idx_beta_remain
                        ]).reshape(-1).tolist()
                        logging.debug(f'idx_vecwup_remain={idx_vecwup_remain}')


                        k_active_list= np.array(k_active_list)[idx_beta_remain].tolist()
                        n_basis_active = len( k_active_list )
                        self.k_active_list = k_active_list
                        alpha_active_expect = (a_alpha_active/b_alpha_active)[idx_alpha_remain]
                        a_alpha_active= np.ones(shape=( n_basis_active ), dtype=dtype) * ( n_basis_active/2 ) + self.a0
                        b_alpha_active= a_alpha_active / alpha_active_expect
                        eta_expect = a_eta/b_eta
                        a_eta = n_basis_active/2 + self.a0
                        b_eta = a_eta / eta_expect
                        beta_active_expect = (a_beta_active/b_beta_active)[idx_beta_remain]
                        a_beta_active= np.ones(shape=( n_basis_active ), dtype=dtype) * ( n_basis_active + 1 )/2 + self.a0
                        b_beta_active= a_beta_active / beta_active_expect
                        a_sigmak_active = a_sigmak_active[idx_beta_remain]
                        b_sigmak_active = b_sigmak_active[idx_beta_remain]
                        mu_zi_list_active= [m[:,idx_alpha_remain] for m in mu_zi_list_active]
                        cov_zi_list_active= [c[ np.ix_(idx_alpha_remain,idx_alpha_remain) ] for c in cov_zi_list_active]
                        mu_vecwup_active= mu_vecwup_active[:,idx_vecwup_remain]
                        cov_vecwup_active= cov_vecwup_active[ np.ix_(idx_vecwup_remain,idx_vecwup_remain) ]
                        w_up_active = mu_vecwup_active.reshape((n_basis_active, n_basis_active)).T
                        mu_zbar_active= mu_zbar_active[:,idx_beta_remain]
                        cov_zbar_active= cov_zbar_active[ np.ix_(idx_beta_remain,idx_beta_remain) ]
                        mu_theta_i_list_active= [m[:,idx_beta_remain] for m in mu_theta_i_list_active]
                        cov_theta_i_list_active= [c[ np.ix_(idx_beta_remain,idx_beta_remain) ] for c in cov_theta_i_list_active]

                        # update the precomputed
                        phi_up_list_active = [
                            np.concatenate([
                                self.kernel_list[ k // n_index_basis ]( self.x_base_index[ [k % n_index_basis] ], zi[0] )
                                for k in self.k_active_list
                            ], axis =0)
                            for i, zi in enumerate(self.x_up_list)
                        ]
                        ppt_list_active = [
                            pi @ pi.T
                            for i,pi in enumerate(phi_up_list_active)
                        ]
                        ppt_list_active_sum = (np.sum(ppt_list_active, axis=0))
                        # ppt_list_active_sum_inv = svd_inverse(ppt_list_active_sum, cond_max=self.cond_num_max, hermitian= True)
                        ypt_list_active = [
                            yi @ phi_up_list_active[i].T
                            for i,yi in enumerate(y_up_list_peudo)
                        ]
                        ypt_list_active_sum = np.sum([
                            ypt_list_active[i]
                            for i,yi in enumerate(y_up_list_peudo)
                        ], axis=0)

                        # update the lower bound
                        lower_bound = get_lower_bound_active(
                            a_white= a_white,
                            b_white= b_white,
                            a_alpha_active= a_alpha_active,
                            b_alpha_active= b_alpha_active,
                            a_beta_active= a_beta_active,
                            b_beta_active= b_beta_active,

                            a_eta= a_eta,
                            b_eta= b_eta,

                            mu_zi_list_active= mu_zi_list_active,
                            cov_zi_list_active= cov_zi_list_active,
                            mu_vecwup_active= mu_vecwup_active,
                            cov_vecwup_active= cov_vecwup_active,
                            mu_zbar_active= mu_zbar_active,
                            cov_zbar_active= cov_zbar_active,
                            y_up_list= y_up_list,
                            yyt_list_sum= yyt_list_sum,
                            ypt_list_active= ypt_list_active,
                            ppt_list_active= ppt_list_active,

                            a0= self.a0,
                            b0= self.b0,
                        )
                        self.n_lbhist_remove_basis_list.append( len(self.lower_bound_history_list) )
                        self.lower_bound_history_list.append(lower_bound)

                        logging.info(f'n_iter={n_iter}, lower_bound={lower_bound:.16e}, sigma={((a_white / b_white)**(-0.5)):.8e}, sigma_k={((a_sigmak_active / b_sigmak_active)**(-0.5))}, \nalpha= {pprint.pformat(a_alpha_active/b_alpha_active, indent=4)}, \nalpha_zbar= {pprint.pformat(a_eta/b_eta, indent=4)}, \nbeta= {pprint.pformat(a_beta_active/b_beta_active, indent=4)}, \nmu_x={mu_zbar_active}, \nw_up={w_up_active}')

                        posterior = get_posterior_probability(
                            z_up_list= mu_zi_list_active,
                            w_up= w_up_active,
                            z_up_bar= mu_zbar_active,
                            alpha= a_alpha_active/b_alpha_active,
                            beta= a_beta_active/b_beta_active,
                            eta= a_eta/b_eta,
                            var_white= (a_white/b_white)**2,
                            y_up_list= y_up_list,
                            phi_up_list= phi_up_list_active,
                            a0= self.a0,
                            b0= self.b0,
                        )
                        self.posterior_history_list.append(posterior)    



                        if dir_debug is not None:
                            self.a_alpha_active = a_alpha_active
                            self.b_alpha_active = b_alpha_active
                            
                            self.a_beta_active = a_beta_active
                            self.b_beta_active = b_beta_active

                            self.a_eta = a_eta
                            self.b_eta = b_eta

                            self.a_white = a_white
                            self.b_white = b_white
                            
                            self.mu_vecwup_active = mu_vecwup_active
                            self.cov_vecwup_active = cov_vecwup_active

                            self.mu_zi_list_active = mu_zi_list_active
                            self.cov_zi_list_active = cov_zi_list_active

                            self.mu_zbar_active = mu_zbar_active
                            self.cov_zbar_active = cov_zbar_active

                            self.mu_theta_i_list_active = mu_theta_i_list_active
                            self.cov_theta_i_list_active = cov_theta_i_list_active

                            self.a_sigmak_active = a_sigmak_active
                            self.b_sigmak_active = b_sigmak_active
                            

                            self.k_active_list = k_active_list                        

                            self.plot_results(
                                path_fig=f'{dir_debug}/bfpca_predict-n_iter{n_iter}-removebasis.pdf',
                            )


            
            # NOTE it could be that after removing redundant basis
            lower_bound_old = copy.deepcopy(lower_bound)

            n_iter += 1


        if dir_debug is not None:
            self.a_alpha_active = a_alpha_active
            self.b_alpha_active = b_alpha_active
            
            self.a_beta_active = a_beta_active
            self.b_beta_active = b_beta_active

            self.a_eta = a_eta
            self.b_eta = b_eta

            self.a_white = a_white
            self.b_white = b_white
            
            self.mu_vecwup_active = mu_vecwup_active
            self.cov_vecwup_active = cov_vecwup_active

            self.mu_zi_list_active = mu_zi_list_active
            self.cov_zi_list_active = cov_zi_list_active

            self.mu_zbar_active = mu_zbar_active
            self.cov_zbar_active = cov_zbar_active

            # parameters specific to the fast algorithm
            self.a_sigmak_active = a_sigmak_active
            self.b_sigmak_active = b_sigmak_active

            self.mu_theta_i_list_active = mu_theta_i_list_active
            self.cov_theta_i_list_active = cov_theta_i_list_active

            self.k_active_list = k_active_list

            self.plot_results(
                path_fig=f'{dir_debug}/bfpca_predict-n_iter{n_iter}.pdf',
            )

            # plot mu_theta_i_list_active_relachange
            plt.figure()
            plt.plot(mu_theta_i_list_active_relachange)
            plt.xlabel('n_iter')
            plt.ylabel('lg max relative change in posterior mean of theta_i')
            plt.savefig(f'{dir_debug}/mu_theta_i_list_active_relachange.pdf')
            plt.close()

            # plot theta_minus_zw_norm_list
            plt.figure()
            plt.plot(theta_minus_zw_norm_list)
            plt.xlabel('n_iter')
            plt.ylabel('sum of norm of (theta_i - z_i w)')
            plt.savefig(f'{dir_debug}/theta_minus_zw_norm_list.pdf')
            plt.close()

            # plot alpha_active_relachange
            plt.figure()
            plt.plot(alpha_active_relachange)
            plt.xlabel('n_iter')
            plt.ylabel('lg max relative change in alpha')
            plt.savefig(f'{dir_debug}/alpha_active_relachange.pdf')
            plt.close()

            # plot beta_active_relachange
            plt.figure()
            plt.plot(beta_active_relachange)
            plt.xlabel('n_iter')
            plt.ylabel('lg max relative change in beta')
            plt.savefig(f'{dir_debug}/beta_active_relachange.pdf')
            plt.close()

        logging.critical(f'BSFDA variational_inference is done! n_iter={n_iter}, lower_bound={lower_bound:.16e}, sigma={((a_white / b_white)**(-0.5)):.8e}, sigma_k={((a_sigmak_active / b_sigmak_active)**(-0.5))}, \nalpha= {pprint.pformat(a_alpha_active/b_alpha_active, indent=4)}, \nalpha_zbar= {pprint.pformat(a_eta/b_eta, indent=4)}, \nbeta= {pprint.pformat(a_beta_active/b_beta_active, indent=4)}, \nmu_x={mu_zbar_active}, \nw_up={w_up_active}')



        self.a_alpha_active = a_alpha_active
        self.b_alpha_active = b_alpha_active
        
        self.a_beta_active = a_beta_active
        self.b_beta_active = b_beta_active

        self.a_eta = a_eta
        self.b_eta = b_eta

        self.a_white = a_white
        self.b_white = b_white
        
        self.mu_vecwup_active = mu_vecwup_active
        self.cov_vecwup_active = cov_vecwup_active

        self.mu_zi_list_active = mu_zi_list_active
        self.cov_zi_list_active = cov_zi_list_active

        self.mu_zbar_active = mu_zbar_active
        self.cov_zbar_active = cov_zbar_active

        self.mu_theta_i_list_active = mu_theta_i_list_active
        self.cov_theta_i_list_active = cov_theta_i_list_active

        self.a_sigmak_active = a_sigmak_active
        self.b_sigmak_active = b_sigmak_active

        self.k_active_list = k_active_list                   

        return self



