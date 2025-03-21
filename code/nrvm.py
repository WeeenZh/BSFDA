import logging
import numpy as np
import matplotlib.pylab as plt
import pprint
import os
import copy
from sklearn.cluster import KMeans
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

# FIXME, HACK, this is a temporary solution to set the threshold for the kernel
KernelPresLatLonTime_threshold = np.array([10, 2e2, 0.2])


class nRVM():
    '''
    yi = ( z_bar + zi ) @ phi_i + ei, zi ~ N(0, diag(beta**(-1))), ei ~ N(0, sigma**2 * I )
    '''
    def __init__(
        self,
        kernel_list,
        max_precision_active = EPS**(-1),
        cond_num_max = None,
        rcond_pinv= 1e-15,
        EPS = EPS,
        ) -> None:
        '''
        max_precision_active:
            maximum allowed precision of the active basis functions, for numerical stability
        cond_num_max:
            maximum allowed condition number of the matrix for solving z_bar, for numerical stability
        '''
        self.cond_num_max= cond_num_max
        self.rcond_pinv= rcond_pinv
        self.kernel_list = kernel_list
        self.max_precision_active = max_precision_active
        self.j_list_similar2active = []
        self.EPS = EPS
        pass

    def phi(
        self, 
        x, 
        x_base_index,
        ):
        '''
        calculate the matrix of basis functions
        ---
        x:
            (n_point, d_index)
        x_base_index:
            (n_base_index, d_index)
        ---
        return:
            (n_base_index * n_kernel, n_point,)
        '''
        return np.concatenate([
            k(x1 = x_base_index, x2 = x)
            for k in self.kernel_list
        ], axis=0)



    def kms_cluster4candidate_bases(
        self,
        x_up_list,
        n_x_base_index = None,
        coef_func_index= None,
        dir_debug= None,
    ):
        '''
        get the basis functions at the candidates of relevacne vectors using kmeans++ clustering
        ---
        x_up_list:
            a list of n_sample (1, n_up_i, d_index) arrays
        coef_func_index:
            multiplier for the number of relevance vector candidates
        n_x_base_index:
            number of candidates basis functions
        ---
        return:
            x_base_index:
                (n_x_base_index, d_index) ndarray
        '''
        # x_up_all = []
        # for x in x_up_list:
        #     x_up_all = x_up_all + x[0].tolist()
        # (n_up_all, d_index)
        x_up_all = np.concatenate(x_up_list, axis=1)[0]


        n_max_x_up_all = 1e4

        if n_x_base_index is None:

            if coef_func_index is not None:
                n_x_base_index = int(np.max([
                    xi.shape[1] for xi in x_up_list
                ]) * coef_func_index)

            else:
                
                # set the number of index set candidates for the relevance vectors
                # the maximum number of index set candidates is 1e4, which is the total number of relevance vectors in the 4d simulation and works weel
                n_x_base_index = min( len(x_up_all), int(n_max_x_up_all) )
        

        if n_x_base_index < len(x_up_all):

            # randomly select 10000 from x_up_all if the number of relevance vectors is larger than 10000, for computational efficiency
            if len(x_up_all) > n_max_x_up_all:
                x_up_all_sample = x_up_all[ np.random.choice( len(x_up_all), size= int(n_max_x_up_all), replace=False ), : ]
            else:
                x_up_all_sample = x_up_all
            
            '''
            km = MiniBatchKMeans(
                n_clusters= n_x_base_index,
                init= 'k-means++',
            )

            km.fit(X= x_up_all_sample)

            # (n_cluster, d_index)
            centers = km.cluster_centers_
            '''
            # use ward linkage to cluster the relevance vectors
            from sklearn.cluster import AgglomerativeClustering
            ward = AgglomerativeClustering(
                n_clusters= n_x_base_index,
                linkage= 'ward',
            )
            n = x_up_all_sample.shape[0]

            m = np.zeros_like(x_up_all_sample[0])
            for x in x_up_all_sample:
                m += x
            m = ( m * (1/n) )[None,:]

            s = np.zeros_like(m)
            for x in x_up_all_sample:
                s += (x[None,:] - m)**2
            s = ( s * (1/n) )**0.5

            x_up_all_sample_normalized = ( x_up_all_sample - m ) * s**(-1)

            ward.fit( x_up_all_sample_normalized )

            x_base_index = np.zeros( shape=(n_x_base_index, x_up_all_sample.shape[1]), dtype=x_up_all_sample.dtype )
            for i in range(n_x_base_index):
                idx = np.where(ward.labels_ == i)[0]
                x = x_up_all_sample_normalized[ idx ]
                nx = x.shape[0]
                
                c = np.zeros_like(x[0])
                for xi in x:
                    c += xi
                c = ( c * (1/nx) )[None,:]

                d = np.zeros( shape=(nx,), dtype=x.dtype )
                for j in range( x.shape[1] ):
                    d += ( x[:,j] - c[0,j] )**2

                idx_c = np.argmin( d )
                x_base_index[i] = x_up_all_sample[ idx[idx_c] ]


            # TODO, to remove
            # x_up_all_sample: blue, x_base_index: red
            n_dims = x_up_all_sample.shape[1]
            n_pairs = n_dims // 2
            remainder = n_dims % 2
            total_plots = n_pairs + remainder

            fig, axes = plt.subplots(1, total_plots, figsize=(5 * total_plots, 5))

            # Ensure axes is iterable
            if total_plots == 1:
                axes = [axes]

            dim_x_y = [(1,2), (0,3)]
            for idx in range(n_pairs):
                ax = axes[idx]
                # dim_x = 2 * idx
                # dim_y = 2 * idx + 1
                dim_x, dim_y = dim_x_y[idx]
                ax.scatter(x_up_all_sample[:, dim_x], x_up_all_sample[:, dim_y], color='blue', alpha=0.5, s=10, label=f'Random {n_max_x_up_all:.0e} of {len(x_up_all):.0e}')
                ax.scatter(x_base_index[:, dim_x], x_base_index[:, dim_y], color='red', marker='x', s=10, label=f'Clustered {n_x_base_index:.0e}')
                ax.set_xlabel(f'Dimension {dim_x}')
                ax.set_ylabel(f'Dimension {dim_y}')
                if idx == 0:
                    ax.legend()

            if dir_debug is not None:
                path_debug_figure = f'{dir_debug}/basis_cluster.pdf'
                plt.savefig(path_debug_figure)
            plt.close()


        else:

            x_base_index = x_up_all


        # sort from the first dimension to the last dimension of the index set ascendingly
        x_base_index_sorted = x_base_index[ np.lexsort(x_base_index[:,::-1].T) ]


        return x_base_index_sorted



    def init_beta(
        self,
        phi_xi_list,
        x_up_list = None,
        y_up_list = None,
    ):
        '''
        # x_base_index:
        #     (n_base, d_index) ndarray of the candidates of index set of the relevance vectors
        ---
        use the given initial parameters or use a heuristic algorithm to initialize the parameters
        '''
        x_up_list= self.x_up_list if x_up_list is None else x_up_list
        y_up_list= self.y_up_list if y_up_list is None else y_up_list
        n_base= self.n_base


        beta = np.ones( shape= (n_base), dtype= x_up_list[0].dtype ) * self.max_precision_active

        # normalize
        # phi_xi_list_normalized = []
        phi_xi_list_scale_inv = []
        for i, xi in enumerate( x_up_list ):
            pxi = phi_xi_list[i]
            s = ( pxi**2 ).sum(axis=1)
            s = (s**(-0.5) )[:,None]
            # pxi_normalized = pxi * s
            # phi_xi_list_normalized.append( pxi_normalized )
            # optimize for memory
            # pxi *= s
            # phi_xi_list_normalized.append( pxi )
            phi_xi_list_scale_inv.append( s )


        # add the most likely relevance vector
        k_init_beta = np.argmax( np.sum(
            [
                ( y_up_list[ i ] @ phi_xi_list[i].T * phi_xi_list_scale_inv[i].T )[0]**2
                for i, xi in enumerate( x_up_list )
            ],
            axis=0
        ) )

        logging.info(f'[param_init] k_init_beta={k_init_beta}')

        beta[k_init_beta] = 1


        return beta


    def get_ln_like(
        self,
        y_up_list,

        z_bar,
        beta,
        sigma,
        phi_xi_list,

        b_up = None,
        d_low = None,
        k2k_active_d_low= None,
        force_float64= False,
        ):
        '''
        likelihood of Yi = Xi @ diag(a)**(-0.5) @ B @ diag(d)**(-0.5) @ Phii + sigma * ei, Xi and ei obey the standard muti variate normal distribution
        ---
        y_up_list:
            a list of n_sample (1, n_index_i, ) arrays
        beta:
            precisions, (1, n_bases)
        b_up:
            identity when None
        d_low:
            identity when None
        sigma:
            standard deviation of white noise
        phi_xi_list:
            Φ
        force_float64:
            force the data type to be float64 for numerical stability
        '''
        dtype0 = y_up_list[0].dtype
        if force_float64:
            dtype = np.float64
            y_up_list = [ yi.astype(dtype) for yi in y_up_list ]
            z_bar = z_bar.astype(dtype)
            beta = beta.astype(dtype)
            sigma = sigma.astype(dtype)
            phi_xi_list = [ phi_xi.astype(dtype) for phi_xi in phi_xi_list ]
            if b_up is not None:
                b_up = b_up.astype(dtype)
            if d_low is not None:
                d_low = d_low.astype(dtype)


        k2k_active_beta= {}
        max_precision_active_dynamic_beta_ = self.get_max_precision_active_dynamic_beta(beta)
        for j,aj in enumerate(beta):
            if aj < max_precision_active_dynamic_beta_:
                k2k_active_beta[j] = len(k2k_active_beta)

        # a_n1 = np.diag( beta.flatten() ** (-1) )
        k_active_beta = sorted(list( k2k_active_beta.keys() ))

        phi_xi_list_active = [
            phi_xi_list[i][k_active_beta,:]
            for i in range(len(y_up_list))
        ]

        if d_low is None:
            if b_up is None:
                bdf_list = [
                    # phi_xi_list[i][k_active_beta,:] 
                    phi_xi_list_active[i]
                    for i, yi in enumerate( y_up_list)
                ]
            else:
                bdf_list = [
                    b_up[ k_active_beta, : ] @ phi_xi_list[i]
                    for i, yi in enumerate( y_up_list)
                ]
        else:
            j_active_d_low = sorted(list( k2k_active_d_low.keys() ))
            d_low_n05_active = d_low[ j_active_d_low ] ** (-0.5)
            
            if b_up is None:
                bdf_list = [
                    d_low_n05_active[:,None] * phi_xi_list[i][j_active_d_low,:]
                    for i, yi in enumerate( y_up_list)
                ]
            else:
                bdf_list = [
                    b_up[ np.ix_( k_active_beta, j_active_d_low, ) ] * d_low_n05_active[None,:] @ phi_xi_list[i][j_active_d_low,:]
                    for i, yi in enumerate( y_up_list)
                ]

        y_up_list_centered = [
            # yi-z_bar[:,k_active_beta] @ phi_xi_list[i][k_active_beta,:]
            yi - z_bar[:,k_active_beta] @ phi_xi_list_active[i]
            for i, yi in enumerate( y_up_list)
        ]

        '''
        # NOTE can be parallelized
        # matrix inverse lemma
        cov_inv_list = [
            woodbury_inverse(
                a_inv = sigma**(-2) * np.identity( n= bdf_list[i].shape[1] ),
                u= bdf_list[i].T,
                c_inv = np.diag( beta[ k_active_beta ] ),
                v= bdf_list[i],
            )
            for i, yi in enumerate( y_up_list)
        ]

        # NOTE can be parallelized
        l_old = np.sum([
            - 1/2 * (
                ( yi.shape[1] ) * np.log( 2 * np.pi) - np.linalg.slogdet( ( cov_inv_list[i] ) )[1] + y_up_list_centered[i] @ cov_inv_list[i] @ y_up_list_centered[i].T
            ) for i, yi in enumerate( y_up_list)
        ])
        '''


        cov_inv_wmi_list = [
            woodbury_inverse(
                a_inv = sigma**(-2),
                u= bdf_list[i].T,
                c_inv = beta[ k_active_beta ],
                v= None,
                return_parts= True,
            )
            for i, yi in enumerate( y_up_list)
        ]
        # cov_inv = a_inv - wmi[0] @ inv(wmi[1]) @ wmi[2]

        # matrix determinant lemma
        cov_inv_logdet_list = [
            woodbury_logdet(
                a_e = sigma**(-2),
                u = cov_inv_wmi_list[i][0],
                w_inv = cov_inv_wmi_list[i][1],
                vt = cov_inv_wmi_list[i][2],
            )
            for i, yi in enumerate( y_up_list)
        ]


        l = np.sum([
            - 1/2 * (
                ( yi.shape[1] ) * np.log( 2 * np.pi) - cov_inv_logdet_list[i] + ( y_up_list_centered[i] @ y_up_list_centered[i].T * sigma**(-2) - y_up_list_centered[i] @ cov_inv_wmi_list[i][0] @ np.linalg.inv( cov_inv_wmi_list[i][1] ) @ cov_inv_wmi_list[i][2] @ y_up_list_centered[i].T )
            ) for i, yi in enumerate( y_up_list)
        ])

        return l.astype( dtype0 )


    def get_update_z_e(
        self,
        z_bar,
        beta,
        sigma,
        eps= EPS,
        active_only=False,
        y_up_list = None,
        phi_xi_list_active = None,
        k_active_beta = None,
        phi_xi_list = None,
        ):
        '''
        calculate the poterior of the latent variables
        ---
        return:
            z_i_e_list:
                E[z_i | *]
            ztz_i_e_list:
                E[z_i.T @ z_i | *]
            z_i_cov_list:
                Cov[z_i | *]
        '''

        y_up_list= self.y_up_list if y_up_list is None else y_up_list
        phi_xi_list= self.phi_xi_list if ( (phi_xi_list_active is None) and (phi_xi_list is None) ) else phi_xi_list

        n_sample = len( y_up_list )

        if k_active_beta is None:
            k2k_active_beta= {}
            max_precision_active_dynamic_beta_ = self.get_max_precision_active_dynamic_beta(beta)
            for j,aj in enumerate(beta):
                if aj < max_precision_active_dynamic_beta_:
                    k2k_active_beta[j] = len(k2k_active_beta)

            k_active_beta = sorted(list(k2k_active_beta.keys()))


        n_index_bases = len(self.x_base_index)
        if phi_xi_list_active is None:
            phi_xi_list_active = [
                phi_xi_list[i][k_active_beta,:]
                # for i, yi in enumerate( y_up_list )
                for i in range(n_sample)
            ]
        
        y_phit_xi_list_active = [
            y_i @ phi_xi_list_active[i].T
            for i, y_i in enumerate( y_up_list )
        ]


        a_active = np.diag( beta[ k_active_beta, ].flatten() )

        phi_phit_xi_list_active = [
            # self.phi_phit_xi_list_func(i=i, j1_list= k_active_beta, j2_list= k_active_beta, phi_xi_list= phi_xi_list)
            phi_xi_list_active[i] @ phi_xi_list_active[i].T
            # for i, y_i in enumerate( y_up_list )
            for i in range(n_sample)
        ]

        mi_list_active = []
        mi_list_active_pinv = []
        # for i, y_i in enumerate( y_up_list ):
        for i in range(n_sample):
            ppxi = phi_phit_xi_list_active[i]
            m_inv = sigma**(-2) * ppxi
            np.fill_diagonal( m_inv, m_inv.diagonal() + beta[ k_active_beta ].flatten() )
            m, m_pinv = svd_inverse(
                m_inv,
                jitter= eps**0.5,
                logging_prefix = f'[mi_list_active] ',
                cond_max= self.cond_num_max,
                rcond_pinv = self.rcond_pinv,
            )
            mi_list_active.append( m )
            mi_list_active_pinv.append( m_pinv )

        z_i_e_list_active = [
            sigma**(-2) * ( y_phit_xi_list_active[i] - z_bar[:,k_active_beta] @ phi_phit_xi_list_active[i] ) @ mi_list_active_pinv[i] 
            for i in range(n_sample)
        ]

        ztz_i_e_list_active = [
            mi_list_active[i] + z_i_e_list_active[i].T @ z_i_e_list_active[i]
            for i in range(n_sample)
        ]

        if not active_only:
            ztz_i_e_list = [
                np.identity( n= beta.shape[0], dtype= y_up_list[0].dtype )
                for i in range(n_sample)
            ]
            z_i_e_list = [
                np.zeros( shape= (1, beta.shape[0]), dtype= y_up_list[0].dtype )
                for i in range(n_sample)
            ]
            z_i_cov_list = [
                np.identity( n= beta.shape[0], dtype= y_up_list[0].dtype )
                for i in range(n_sample)
            ]
            for i in range(n_sample):
                z_i_e_list[i][ 0, k_active_beta ] = z_i_e_list_active[i][ 0, : ]

                ztz_i_e_list[i][ np.ix_( k_active_beta, k_active_beta ) ] = ztz_i_e_list_active[i]

                z_i_cov_list[i][ np.ix_( k_active_beta, k_active_beta ) ] = mi_list_active[i]


            return z_i_e_list, ztz_i_e_list, z_i_cov_list
        else:
            return z_i_e_list_active, ztz_i_e_list_active, mi_list_active
            
    # def phi_phit_xi_list_func(
    #     self,
    #     i,
    #     j1_list,
    #     j2_list,
    #     phi_xi_list = None,
    #     ):
    #     '''
    #     return the selected basis functions for the i-th sample
    #     '''
    #     phi_xi_list = self.phi_xi_list if phi_xi_list is None else phi_xi_list
    #     return phi_xi_list[i][j1_list,:] @ phi_xi_list[i][j2_list,:].T


    def init_params(
        self,
        coef_func_index,
        x_base_index,
        ratio_init_var_noise,
        sigma_init,
        compute_loglike= True,
        dir_debug= None,
        batch_size= None,
    ):
        x_up_list= self.x_up_list
        y_up_list= self.y_up_list

        
        # (n_x_base_index, d_index)
        self.x_base_index = self.kms_cluster4candidate_bases(
            coef_func_index=coef_func_index,
            x_up_list= x_up_list,
            dir_debug= dir_debug,
        ) if ( x_base_index is None) else x_base_index

        logging.info(f'x_base_index.shape= {self.x_base_index.shape}, self.x_base_index= {pprint.pformat( self.x_base_index, indent= 4)}')

        # Store full phi_xi_list instead of function
        logging.info(f'calculate phi_xi_list')
        # HACK to avoid memory error, assuming later x_up_list is larger and needs more memory cache, does not change the result
        # self.phi_xi_list = [
        #     self.phi(xi[0,:], x_base_index=self.x_base_index) 
        #     for xi in x_up_list
        # ]
        self.phi_xi_list = [None for xi in x_up_list]
        _ = list(range(len(x_up_list)))
        _.reverse()
        for i in _:
            self.phi_xi_list[i] = self.phi(x_up_list[i][0,:], x_base_index=self.x_base_index)
        phi_xi_list = self.phi_xi_list

        # def phi_xi_list_func(
        #     i,
        #     k_list='all',
        #     phi_xi_list = phi_xi_list,
        # ):
        #     '''
        #     return the selected basis functions for the i-th sample
        #     '''
        #     if k_list=='all':
        #         return phi_xi_list[i]
        #     else:
        #         return phi_xi_list[i][ k_list, : ]

        # # list of n_sample (n_base, n_index_i) arrays
        # self.phi_xi_list_func = phi_xi_list_func
        # self.n_base = phi_xi_list_func(0).shape[0]
        self.n_base = self.phi_xi_list[0].shape[0]

        # mean of the latent variables  
        self.z_bar = np.zeros(shape=(1, self.n_base), dtype= y_up_list[0].dtype)


        # precompute
        # self.phi_phit_xi_list_func = lambda i, j1_list, j2_list, phi_xi_list= self.phi_xi_list : phi_xi_list[i][j1_list,:] @ phi_xi_list[i][j2_list,:].T

        if batch_size is not None:
            # use first batch to initialize
            x_up_list_batch = []
            y_up_list_batch = []
            phi_xi_list_batch = []

            i = 0
            idx = list(range( y_up_list[i].shape[1] ))
            iter_per_epoch_ = int( np.ceil( len(idx) / batch_size ) )
            n = 0
            phi_xi_list_batch.append( phi_xi_list[i][:, n*batch_size : (n+1)*batch_size] )
            x_up_list_batch.append( x_up_list[i][:, n*batch_size : (n+1)*batch_size] )
            y_up_list_batch.append( y_up_list[i][:, n*batch_size : (n+1)*batch_size] )

        else:
            phi_xi_list_batch = phi_xi_list
            x_up_list_batch = x_up_list
            y_up_list_batch = y_up_list
        
        # initialize beta
        logging.info(f'initialize beta')
        self.beta = self.init_beta(
            phi_xi_list= phi_xi_list_batch,
            x_up_list= x_up_list_batch,
            y_up_list= y_up_list_batch,
        )

        self.k2k_active_beta = OrderedDict()
        max_precision_active_dynamic_beta_ = self.get_max_precision_active_dynamic_beta(self.beta)
        for i, ei in enumerate( np.argwhere( self.beta < max_precision_active_dynamic_beta_ ) ):
            self.k2k_active_beta[ int(ei) ] = i

        logging.info(f'[param_init] j_beta_active={ pprint.pformat( self.zip_beta( beta= self.beta, ), indent= 4) }')


        # assign a sensable variance to the white noise
        dtype = y_up_list_batch[0].dtype
        if (sigma_init is None):

            logging.info(f'initialize sigma')
            self.sigma = ( ( np.dtype(dtype).type(ratio_init_var_noise) * np.sum([ yp@yp.T for yp in y_up_list_batch]) / np.sum([ len(yp) for yp in y_up_list_batch ]).astype(dtype) )**0.5 ).astype(dtype)

            # check logging level
            if logging.getLogger().getEffectiveLevel() <= logging.INFO:
                llkh = self.get_ln_like( y_up_list = y_up_list_batch, z_bar=self.z_bar, beta = self.beta, sigma = self.sigma, phi_xi_list= phi_xi_list_batch ) if compute_loglike else -np.nan
                logging.info(f'[param_init] sigma={self.sigma :.8e}, log likelihood= {llkh :.8e}')

            # initialize sigma using em
            logging.info(f'update initial sigma using E in EM')
            z_i_e_list_active, ztz_i_e_list_active, z_i_cov_list_active = self.get_update_z_e(
                z_bar = self.z_bar,
                beta= self.beta,
                sigma= self.sigma,
                active_only= True,
                y_up_list = y_up_list_batch,
                phi_xi_list= phi_xi_list_batch,
            )
            
            logging.info(f'update initial sigma using M in EM')
            self.sigma = self.get_mstep_update_sigma(
                z_bar = self.z_bar,
                beta= self.beta,
            
                z_i_e_list_active= z_i_e_list_active,
                ztz_i_e_list_active= ztz_i_e_list_active,
                z_i_cov_list_active= z_i_cov_list_active,

                y_up_list= y_up_list_batch,
                phi_xi_list= phi_xi_list_batch,
            )
            if logging.getLogger().getEffectiveLevel() <= logging.INFO:
                llkh = self.get_ln_like( y_up_list = y_up_list_batch, z_bar=self.z_bar, beta = self.beta, sigma = self.sigma, phi_xi_list=phi_xi_list_batch ) if compute_loglike else -np.nan
                logging.info(f'[param_update] sigma={self.sigma :.8e}, log likelihood= {llkh :.8e}')

        else:
            self.sigma = np.dtype(dtype).type(sigma_init)

            if logging.getLogger().getEffectiveLevel() <= logging.INFO:
                llkh = self.get_ln_like( y_up_list = y_up_list_batch, z_bar=self.z_bar, beta = self.beta, sigma = self.sigma, phi_xi_list=phi_xi_list_batch ) if compute_loglike else -np.nan
                logging.info(f'[param_init] sigma={self.sigma :.8e}, log likelihood= {llkh :.8e}')





    def get_max_precision_active_dynamic_beta(
        self,
        beta,
        ):
        '''
        this currently does nothing but return self.max_precision_active
        '''
        return self.max_precision_active


    def get_mstep_update_sigma(
        self,
        z_bar,
        beta,

        z_i_e_list_active,
        ztz_i_e_list_active,
        z_i_cov_list_active,
        y_up_list = None,
        phi_xi_list_active = None,
        k_active_beta = None,
        phi_xi_list = None,
        ):
        '''
        m step for sigma
        '''
        y_up_list= self.y_up_list if y_up_list is None else y_up_list
        phi_xi_list = self.phi_xi_list if phi_xi_list is None else phi_xi_list

        dtype = y_up_list[0].dtype

        # phi_phit_xi_list_func = self.phi_phit_xi_list_func

        if k_active_beta is None:
            k2k_active_beta= {}
            max_precision_active_dynamic_beta_ = self.get_max_precision_active_dynamic_beta(beta)
            for j,aj in enumerate(beta):
                if aj < max_precision_active_dynamic_beta_:
                    k2k_active_beta[j] = len(k2k_active_beta)
            k_active_beta = sorted(tuple(k2k_active_beta.keys()))

        if phi_xi_list_active is None:
            phi_xi_list_active = [
                phi_xi_list[i][k_active_beta,:]
                # for i, yi in enumerate( y_up_list )
                for i in range(len(y_up_list))
            ]

        # phi_phit_xi_list_active = [
        #     phi_phit_xi_list_func(i=i, j1_list= k_active_beta, j2_list= k_active_beta)
        #     for i, yi in enumerate( y_up_list )
        # ]

        y_phit_list_centered_active = [
            # yi - (z_bar[:, k_active_beta] @ phi_xi_list[i][k_active_beta,:])
            yi - (z_bar[:, k_active_beta] @ phi_xi_list_active[i])
            for i, yi in enumerate( y_up_list )
        ]
        
        # O( p * ( ni**2 + ni * k + k**2 ) )
        # sigma_new_sq = (np.float64(np.sum([
        #     yi.shape[1]
        #     for i, yi in enumerate( y_up_list )
        # ]))**(-1)).astype( dtype ) * \
        #     np.sum([
        #         y_phit_list_centered_active[i] @ y_phit_list_centered_active[i].T - 2 * y_phit_list_centered_active[i] @ phi_xi_list[i][k_active_beta,:].T @ z_i_e_list_active[i].T + \
        #             np.sum( ztz_i_e_list_active[i] * phi_phit_xi_list_active[i] )
        #         for i, yi in enumerate( y_up_list )
        #     ])
        # change to the following to avoid numerical instability, keep it positive
        # O( k*ni*p )
        sigma_new_sq = (np.float64(np.sum([
            yi.shape[1]
            for i, yi in enumerate( y_up_list )
        ]))**(-1)).astype( dtype ) * \
            np.sum([
                # ( ( y_phit_list_centered_active[i] - z_i_e_list_active[i] @ phi_xi_list[i][k_active_beta,:] )**2 ).sum()
                ( ( y_phit_list_centered_active[i] - z_i_e_list_active[i] @ phi_xi_list_active[i] )**2 ).sum() + ( z_i_cov_list_active[i] * ( phi_xi_list_active[i] @ phi_xi_list_active[i].T ) ).sum()
                for i, yi in enumerate( y_up_list )
            ])
        
        sigma_new = ( sigma_new_sq**(0.5) ).astype( dtype )

        return sigma_new


    def zip_beta(
        self,
        beta,
    ):
        x_base_index_all_kernel = np.array( self.x_base_index.tolist() * len(self.kernel_list), dtype= self.x_base_index.dtype )

        k2k_active_beta= {}
        max_precision_active_dynamic_beta_ = self.get_max_precision_active_dynamic_beta(beta)
        for j,aj in enumerate(beta):
            if aj < max_precision_active_dynamic_beta_:
                k2k_active_beta[j] = len(k2k_active_beta)

        jas= sorted(list( k2k_active_beta.keys() ))
        j_beta_x_active = sorted(
            zip(
                jas,
                beta[ jas ],
                x_base_index_all_kernel[ jas ]
            ),
            key= lambda e:e[1],
        )
        return j_beta_x_active


    def fit(
        self,
        x_up_list,
        y_up_list,
        # initialization
        coef_func_index= None,
        x_base_index=None,
        ratio_init_var_noise=0.1,
        sigma_init=None,
        # fast fitting
        n_iter_max_fast=100,
        n_patience= 10,
        tol= None,
        sigma_is_fixed= False,
        sim_threshold = 0.999,
        n_basis_add_serial = 1,
        # iter_per_epoch= 1,
        batch_size= None,
        compute_loglike= True,
        # em update
        tol_plateau= None,
        # trim
        n_log= 1,
        dir_debug= None,        
    ):
        self.max_precision_active = np.dtype(x_up_list[0].dtype).type(self.max_precision_active)

        self.x_up_list= x_up_list
        self.y_up_list= y_up_list

        tol_plateau = 1e-2 * len(y_up_list) if tol_plateau is None else tol_plateau

        self.init_params(
            coef_func_index= coef_func_index,
            x_base_index= x_base_index,
            ratio_init_var_noise= ratio_init_var_noise,
            sigma_init= sigma_init,
            compute_loglike= compute_loglike,
            dir_debug= dir_debug,
            batch_size= batch_size,
        )
        self.z_bar, self.beta, self.sigma = self.get_update_fast(
            z_bar= self.z_bar,
            beta= self.beta,
            sigma= self.sigma,

            n_iter_max= n_iter_max_fast,
            n_patience= n_patience,
            tol= tol,
            tol_plateau= tol_plateau,
            sigma_is_fixed= sigma_is_fixed,
            n_basis_add_serial = n_basis_add_serial,
            dir_debug= dir_debug if (dir_debug is None) else f'{dir_debug}/fast' ,
            
            n_log= n_log,
            sim_threshold = sim_threshold,
            # iter_per_epoch= iter_per_epoch,
            batch_size= batch_size,
            compute_loglike= compute_loglike,
        )

        if dir_debug is not None:
            self.plot_results(
                z_bar= self.z_bar,
                beta= self.beta,
                sigma= self.sigma,

                path_fig= f'{dir_debug}/final.pdf',
                compute_loglike = compute_loglike,
                )
            self.dump( path_dump= f'{dir_debug}/nrvm.pkl')


        logging.critical(f'nRVM fitting is done!')

        return self

    def dump(
            self, 
            path_dump,
            without_data= True,
            ):
        '''
        use dill to dump the object, but it can be loaded by pickle
        '''
        if without_data:
            cache_ = {}
            # remove the data to reduce the size of the file
            attributes_to_exclude = [
                'x_up_list',
                'y_up_list',
                'phi_xi_list',
            ]

            for attr in attributes_to_exclude:
                if hasattr(self, attr):
                    cache_[attr] = getattr(self, attr)
                    delattr(self, attr)

            with open(path_dump, 'wb') as f:
                dill.dump(
                    obj= self,
                    file= f,
                )

            # restore the data
            for k,v in cache_.items():
                setattr(self, k, v)

        else:
            with open(path_dump, 'wb') as f:
                dill.dump(
                    obj= self,
                    file= f,
                )

    
    def k_list_get_fm(
        self,
        j_list,
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
                for j in j_list
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
                for j in j_list
            ]

            fm_j = PresLatLonTimeFunctionMixture( lat_lon_pres_time_function_list= f_j_list )

        return fm_j
    

    def if_similar(
        self,
        func_mix,
        j_list2,
        sim_threshold,
        KernelPresLatLonTime_threshold = KernelPresLatLonTime_threshold,
        ):
        '''
        check if a new list of basis functions is similar to the active basis functions
        ---
        func_mix:
            the mixture of functions
        '''

        is_not_similar = None
        if isinstance( self.kernel_list[0], RBFKernel2 ):
            sim2active = func_mix.cosine_similar2subspace(
                rbfm= self.k_list_get_fm( j_list= j_list2 ),
            )
            is_not_similar = sim2active < sim_threshold
        elif isinstance( self.kernel_list[0], KernelPresLatLonTime ):
            is_not_similar = not func_mix.check_if_similar(
                lat_lon_pres_time_function= self.k_list_get_fm( j_list= j_list2 ).lat_lon_pres_time_function_list[0],
                threshold = KernelPresLatLonTime_threshold,
            )
            # HACK TODO 
            sim2active = None
        else:
            raise NotImplementedError

        is_similar = not is_not_similar
        
        return is_similar, sim2active


    def get_update_fast(
        self,
        z_bar,
        beta,
        sigma,

        n_iter_max=100,
        n_patience= 10,
        tol= None,
        tol_plateau= None,
        sigma_is_fixed= False,

        n_basis_add_serial = 1,


        sim_threshold = 0.999,
        n_log= 1,

        dir_debug= None,

        KernelPresLatLonTime_threshold = KernelPresLatLonTime_threshold,

        # iter_per_epoch= 1,
        batch_size = None,

        compute_loglike= True,
        ):
        '''
        compute updates of precisions using direct differentiation with approximations
        ---
        n_iter_max:
            a positive integer
        n_basis_add_serial:
            number of basis functions to add at once
        iter_per_epoch:
            a positive float, the factor to multiply the number of observations in the optimization
        batch_size:
            a positive integer, the size of the batch of each functional instance for stochastic optimization, if None, no stochastic optimization
        '''
        # x_up_list= self.x_up_list.copy()
        y_up_list= self.y_up_list.copy()
        n_sample = len( y_up_list )

        tol= 1e-5 * n_sample if (tol is None) else tol
        tol_plateau= 1e-2 * n_sample if (tol_plateau is None) else tol_plateau




        # j2j_active= {}
        j_active_set = set()
        max_precision_active_dynamic_beta_ = self.get_max_precision_active_dynamic_beta(beta)
        for j,aj in enumerate(beta):
            if aj < max_precision_active_dynamic_beta_:
                # j2j_active[j] = len(j2j_active)
                j_active_set.add(j)
        # j_active_slice = sorted(tuple( j2j_active.keys() ))
        j_active_slice = sorted(list( j_active_set ))


        # y_up_list_centered = [
        #     yi-z_bar[:,j_active_slice] @ self.phi_xi_list[i][j_active_slice,:]
        #     for i, yi in enumerate( y_up_list)
        # ]


        if (dir_debug is not None) and (not os.path.exists( dir_debug )):
            os.makedirs( dir_debug )

        phi_xi_list = self.phi_xi_list
        n_base = phi_xi_list[0].shape[0]


        s_up = np.zeros(shape= (n_base, n_sample), dtype= y_up_list[0].dtype)
        q_up = np.zeros(shape= (n_base, n_sample), dtype= y_up_list[0].dtype)
        theta = np.zeros_like( beta )

        # if beta changes, the covariances needs to be updated
        # j_active_slice = list(j2j_active.keys())


        converged = False
        on_plateau= False
        n_iter = 0
        # ln_like = self.get_ln_like( y_up_list = y_up_list, z_bar=z_bar, beta = beta, sigma = sigma, phi_xi_list=self.phi_xi_list ) if compute_loglike else -np.nan
        ln_like = - np.inf if compute_loglike else -np.nan

        # here beta is directly next to basis functions thus it like d_low
        logging.info(f'[precision_update_fast] log likelihood= {ln_like:.8e}, j_beta_active={ pprint.pformat( self.zip_beta( beta= beta ), indent= 4) }')


        n_non_improve = 0
        best_return = {
            'beta': beta.copy(), 
            'z_bar': z_bar.copy(), 
            # 'j2j_active': j2j_active.copy(),
            'j_active_set': j_active_set.copy(),
            'sigma': copy.deepcopy(sigma),
            'ln_like': ln_like,
        }
        # history of deletion of the basis functions
        j_list_deleted = []
        # NOTE self.j_list_similar2active need to be updated when there is an active basis functions deleted because the similarity could decrease
        self.j_list_similar2active = []

        n_basis_to_add = 0
        j_candidate_list = []




        while ( not converged ):

            n_iter += 1

            # list of stochastic indices
            # if iter_per_epoch > 1:
            if batch_size is not None:
                phi_xi_list_batch = []
                y_up_list_batch = []
                y_up_list_centered_batch = []

                for i in range( n_sample ):
                    idx = list(range( y_up_list[i].shape[1] ))
                    iter_per_epoch_ = int( np.ceil( len(idx) / batch_size ) )
                    n = n_iter % iter_per_epoch_
                    phi_xi_list_batch.append( phi_xi_list[i][:, n*batch_size : (n+1)*batch_size] )
                    y_up_list_batch.append( y_up_list[i][:, n*batch_size : (n+1)*batch_size] )

                    y_up_list_centered_batch.append( y_up_list_batch[i] - z_bar[:,j_active_slice] @ phi_xi_list_batch[i][j_active_slice,:] )
            else:
                phi_xi_list_batch = phi_xi_list
                y_up_list_batch = y_up_list
                y_up_list_centered_batch = [
                    yi-z_bar[:,j_active_slice] @ phi_xi_list[i][j_active_slice,:]
                    for i, yi in enumerate( y_up_list)
                ]


            '''
            # c_up_inv_xi_list = [
            #     np.linalg.inv( sigma**2 * np.identity(n= xp.shape[1]) + ( phi_xi_list_func(p, k_list=j_active_slice) ).T @ np.diag( beta[ j_active_slice ] ** (-1) ) @ ( phi_xi_list_func(p, k_list=j_active_slice) ) )
            #     for p, xp in enumerate( x_up_list )
            # ]
            # woodbury_inverse
            c_up_inv_xi_list = [
                woodbury_inverse(
                    a_inv = sigma**(-2),
                    u= ( phi_xi_list[p][j_active_slice,:] ).T,
                    c_inv = np.diag( beta[ j_active_slice ] ),
                    v= None,
                )
                for p, xp in enumerate( x_up_list )
            ]
            '''
            c_up_inv_xi_wmi_list = [
                woodbury_inverse(
                    a_inv = sigma**(-2),
                    # u= ( phi_xi_list[p][j_active_slice,stci_list[p]] ).T,
                    u= ( phi_xi_list_batch[p][j_active_slice,:] ).T,
                    c_inv = beta[ j_active_slice ],
                    v= None,
                    return_parts= True,
                )
                for p in range( n_sample )
            ]
            for p in range( n_sample ):
                c_up_inv_xi_wmi_list[p] = tuple( list(c_up_inv_xi_wmi_list[p]) + [ np.linalg.inv( c_up_inv_xi_wmi_list[p][1] ) ] )
            # c_up_inv_xi = a_inv - wmi[0] @ inv(wmi[1]) @ wmi[2], a_inv = sigma**(-2) * I, wmi[3] = inv(wmi[1])


            j_list = list( range( len( beta )) )
            np.random.shuffle( j_list )

            def jb_get_tj_s(
                j,
                beta,
                get_max_precision_active_dynamic_beta = self.get_max_precision_active_dynamic_beta,
                # phi_xi_list = phi_xi_list,
                phi_xi_list_batch = phi_xi_list_batch,
                n_sample = n_sample,
                sigma = sigma,
                # y_up_list_centered = y_up_list_centered,
                y_up_list_centered_batch = y_up_list_centered_batch,
                c_up_inv_xi_wmi_list = c_up_inv_xi_wmi_list,
                *args,
                **kwargs,
                ):
                pij_list = [
                    phi_xi_list_batch[p][[j],:]  # Changed indexing
                    for p in range(n_sample)
                ]
                q_up_j = np.array([
                    ( pij_list[p] @ ( sigma**(-2) * y_up_list_centered_batch[p].T - ( y_up_list_centered_batch[p] @ c_up_inv_xi_wmi_list[p][2].T @ c_up_inv_xi_wmi_list[p][3].T @ c_up_inv_xi_wmi_list[p][0].T ).T ) )[0,0]
                    for p in range( n_sample )
                ])
                s_up_j = np.array([
                    ( pij_list[p] @ ( sigma**(-2) * pij_list[p].T - ( pij_list[p] @ c_up_inv_xi_wmi_list[p][2].T @ c_up_inv_xi_wmi_list[p][3].T @ c_up_inv_xi_wmi_list[p][0].T ).T ) )[0,0]
                    for p in range( n_sample )
                ])

                if beta[j] >= get_max_precision_active_dynamic_beta(beta):
                    q = q_up_j
                    s = s_up_j
                else:
                    q = beta[j] * q_up_j / ( beta[j] - s_up_j  )
                    s = beta[j] * s_up_j / ( beta[j] - s_up_j  )
                theta_j = np.sum( q**2 - s )

                return theta_j, s


            # order the basis functions based on theta and similarity
            if (on_plateau or not compute_loglike) and n_basis_to_add <= 0:

                # update the order of the basis functions to add

                j2sim2active = {}

                fm_active = self.k_list_get_fm( j_list= list( j_active_set ) )


                for j in j_list:
                    
                    # skip the basis functions that are already similar to the active ones
                    if ( (j in self.j_list_similar2active) or (j in j_active_set) ):
                        continue

                    # cosine similarity between the new basis function and the optimized ones
                    # reuse fm_active to save time from orthornomalization
                    is_similar, sim2active = self.if_similar(
                        func_mix= fm_active,
                        j_list2= [j,],
                        sim_threshold= sim_threshold,
                        KernelPresLatLonTime_threshold = KernelPresLatLonTime_threshold,
                    )
                    is_not_similar = not is_similar
                    logging.debug(f'j={j}, sim2active={sim2active}')


                    if is_not_similar:

                        j2sim2active[j] = sim2active
                        # q_up_j = np.array([ ( phi_xi_list_func(p, k_list=[j]) @ c_up_inv_xi_list[ p ] @ y_up_list_centered[ p ].T )[0,0] for p in range( n_sample )])
                        # s_up_j = np.array([ ( phi_xi_list_func(p, k_list=[j]) @ c_up_inv_xi_list[ p ] @ phi_xi_list_func(p, k_list=[j]).T )[0,0] for p in range( n_sample )])
                        # optimized

                        theta[j], s = jb_get_tj_s(j = j, beta= beta)
                    
                    else:
                        self.j_list_similar2active.append(j)

                if len(j2sim2active) > 0:
                    theta_j_candidate_list = [
                        (theta[j],j) for j in j2sim2active
                    ]

                    theta_j_candidate_list = sorted(theta_j_candidate_list, reverse=True)
                    j_candidate_list = np.array(theta_j_candidate_list)
                    j_candidate_list = j_candidate_list[ j_candidate_list[:,0] > 0 ][ :,1 ].astype(int).tolist()

                    logging.debug(f'[sorting new bases] theta_j_candidate_list={pprint.pformat(theta_j_candidate_list)}')

                    n_basis_to_add = n_basis_add_serial


            if ( n_basis_to_add > 0 ) and ( len(j_candidate_list) > 0):

                fm_active = self.k_list_get_fm( j_list= list( j_active_set ) )

                for j in j_candidate_list:

                    logging.debug(f'consider the basis, j={j}, theta[j]={theta[j]}')

                    if theta[j] > 0:
                        
                        if n_basis_to_add == n_basis_add_serial:
                            # the active basis functions are the same, similarity has been checked
                            is_not_similar = True
                        else:
                            # the active basis functions are updated, check similarity
                            is_similar, sim2active = self.if_similar(
                                func_mix= fm_active,
                                j_list2= [j,],
                                sim_threshold= sim_threshold,
                                KernelPresLatLonTime_threshold = KernelPresLatLonTime_threshold,
                            )
                            is_not_similar = not is_similar
                        
                        if is_not_similar:
                            # add this basis function
                            # calculate the updated theta_j and s with the updated beta
                            if n_basis_to_add < n_basis_add_serial:
                                theta_j, s = jb_get_tj_s(j = j, beta= beta)
                            else:
                                theta_j, s = theta[j], s

                            if theta_j > 0:

                                theta[j] = theta_j

                                beta[j] = np.sum( s**2 ) / theta[j]

                                logging.debug(f'add the basis, beta[j]={beta[j]}')

                                # j2j_active[j] = len(j2j_active)
                                j_active_set.add(j)

                                n_basis_to_add -= 1

                                j_candidate_list.remove(j)

                                # add one basis at one iteration, need to update the other variables before adding the next one
                                break
                            


            j_list_active = list( j_active_set )
            np.random.shuffle(j_list_active)

            # c_up_inv_xi_list_2 = None
            # c_up_inv_xi_list_ = None
            # update the active dimensions
            for j in j_list_active:
                # q_up_j = np.array([ ( phi_xi_list_func(p, k_list=[j]) @ c_up_inv_xi_list[ p ] @ y_up_list_centered[ p ].T )[0,0] for p in range( n_sample )])
                # s_up_j = np.array([ ( phi_xi_list_func(p, k_list=[j]) @ c_up_inv_xi_list[ p ] @ phi_xi_list_func(p, k_list=[j]).T )[0,0] for p in range( n_sample )])
                # optimized
                theta[j], s = jb_get_tj_s(
                    j = j, 
                    beta= beta,
                    y_up_list_centered_batch = y_up_list_centered_batch,
                    c_up_inv_xi_wmi_list = c_up_inv_xi_wmi_list,
                    sigma = sigma,
                    # c_up_inv_xi_list_2 = c_up_inv_xi_list_2,
                    # c_up_inv_xi_list_ = c_up_inv_xi_list_,
                    )
                if theta[j] > 0:

                    max_precision_active_dynamic_beta_ = self.get_max_precision_active_dynamic_beta(beta)
                    if beta[j] < max_precision_active_dynamic_beta_:

                        beta[j] = np.sum( s**2 ) / theta[j]

                        logging.debug(f're estimated beta[j]={beta[j]}')

                else:
                    
                    logging.debug('theta[j] <= 0')
                    
                    max_precision_active_dynamic_beta_ = self.get_max_precision_active_dynamic_beta(beta)
                    if beta[j] < max_precision_active_dynamic_beta_:

                        if len( j_active_set ) > 1:

                            logging.debug('delete the basis')
                            j_list_deleted.append(j)

                            beta[j] = max_precision_active_dynamic_beta_

                            # j_active = j2j_active[j]
                            # # update the indices that are active after deleting the current one
                            # j2j_active.pop(j)
                            # for k,v in j2j_active.items():
                            #     if v > j_active:
                            #         j2j_active[k] = v-1
                            j_active_set.remove(j)
                        
                            self.j_list_similar2active = []

                        else:
                            pass
                            logging.debug('skip deleting the basis, because this is the only basis')



                j_active_slice = sorted(list( j_active_set ))

                if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
                    ln_like = self.get_ln_like( y_up_list = y_up_list_batch, z_bar=z_bar, beta = beta, sigma = sigma, phi_xi_list=phi_xi_list_batch ) if compute_loglike else -np.nan
                    logging.debug(f'[precision_update_fast] n_iter= {n_iter}, n_non_improve= {n_non_improve}, log likelihood= {ln_like :.8e}, sigma={sigma:.8e}, j_beta_active={ pprint.pformat( self.zip_beta( beta= beta), indent= 4) } \nz_bar[:, j_active_slice]={z_bar[:, j_active_slice]}')

                phi_xi_list_batch_active = [ phi_xi_list_batch[p][j_active_slice,:] for p in range( n_sample ) ]
                # update white noise
                z_i_e_list_active, ztz_i_e_list_active, z_i_cov_list_active = self.get_update_z_e(
                    beta= beta,
                    sigma= sigma,
                    z_bar = z_bar,
                    active_only= True,
                    # phi_phit_xi_list_active = [
                    #     self.phi_phit_xi_list_func(i=i, j1_list= j_active_slice, j2_list= j_active_slice, phi_xi_list= phi_xi_list_batch)
                    #     for i, y_i in enumerate( y_up_list )
                    # ],
                    y_up_list= y_up_list_batch,
                    # phi_xi_list= phi_xi_list_batch,
                    phi_xi_list_active= phi_xi_list_batch_active,
                    k_active_beta= j_active_slice,
                )
                if not sigma_is_fixed:
                    sigma = self.get_mstep_update_sigma(
                        beta= beta,
                        z_bar = z_bar,
                        z_i_e_list_active= z_i_e_list_active,
                        ztz_i_e_list_active= ztz_i_e_list_active,
                        z_i_cov_list_active= z_i_cov_list_active,
                        y_up_list= y_up_list_batch,
                        # phi_xi_list= phi_xi_list_batch,
                        phi_xi_list_active= phi_xi_list_batch_active,
                        k_active_beta= j_active_slice,
                    )
                if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
                    ln_like = self.get_ln_like( y_up_list = y_up_list_batch, z_bar=z_bar, beta = beta, sigma = sigma, phi_xi_list=phi_xi_list_batch ) if compute_loglike else -np.nan
                    logging.debug(f'[precision_update_fast] n_iter= {n_iter}, n_non_improve= {n_non_improve}, log likelihood= {ln_like :.8e}, sigma={sigma:.8e}, j_beta_active={ pprint.pformat( self.zip_beta( beta= beta), indent= 4) } \nz_bar[:, j_active_slice]={z_bar[:, j_active_slice]}')

                # NOTE can be parallelized
                # matrix inverse lemma
                # update s_up and q_up
                '''
                c_up_inv_xi_list = [
                    woodbury_inverse(
                        a_inv= sigma**(-2),
                        # u= phi_xi_list_func(p, k_list=j_active_slice).T,
                        u= phi_xi_list_batch_active[p].T,
                        c_inv= np.diag( beta[ j_active_slice ] ),
                        v= None
                    )
                    for p in range( n_sample )
                ]
                c_up_inv_xi_list_ = [
                    woodbury_inverse(
                        a_inv= sigma**(-2) * np.identity(n= y_up_list[p].shape[1]),
                        # u= phi_xi_list_func(p, k_list=j_active_slice).T,
                        u= phi_xi_list_batch_active[p].T,
                        c_inv= np.diag( beta[ j_active_slice ] ),
                        # v= phi_xi_list_func(p, k_list=j_active_slice)
                        v= phi_xi_list_batch_active[p]
                    )
                    for p in range( n_sample )
                ]                
                '''
                # c_up_inv_xi_wmi_list = [
                #     woodbury_inverse(
                #         a_inv = sigma**(-2),
                #         # u= ( phi_xi_list[p][j_active_slice,:][:,stci_list[p]] ).T,
                #         # u= ( phi_xi_list_batch[p][j_active_slice,:] ).T,
                #         u= ( phi_xi_list_batch_active[p] ).T,
                #         c_inv = np.diag( beta[ j_active_slice ] ),
                #         v= None,
                #         return_parts= True,
                #     )
                #     # for p, xp in enumerate( x_up_list )
                #     for p in range( n_sample )
                # ]
                c_up_inv_xi_wmi_list = []
                # c_inv_ = np.diag( beta[ j_active_slice ] )
                # c_inv_ = beta[ j_active_slice ]
                for p in range( n_sample ):
                    a_= woodbury_inverse(
                            a_inv = sigma**(-2),
                            u= ( phi_xi_list_batch_active[p] ).T,
                            c_inv = beta[ j_active_slice ],
                            v= None,
                            return_parts= True,
                        )
                    c_up_inv_xi_wmi_list.append(a_)
                # for p, xp in enumerate( x_up_list ):
                for p in range( n_sample ):
                    c_up_inv_xi_wmi_list[p] = tuple( list(c_up_inv_xi_wmi_list[p]) + [ np.linalg.inv( c_up_inv_xi_wmi_list[p][1] ) ] )
                # c_up_inv_xi_list_2 = [
                #     sigma**(-2) * np.identity(n= y_up_list[p].shape[1]) - c_up_inv_xi_wmi_list[p][0] @ c_up_inv_xi_wmi_list[p][3] @ c_up_inv_xi_wmi_list[p][2]
                #     for p in range( n_sample )
                # ]
                # c_up_inv_xi = a_inv - wmi[0] @ inv(wmi[1]) @ wmi[2], a_inv = sigma**(-2) * I, wmi[3] = inv(wmi[1])
            
                # NOTE z_bar is reset to zero and then updated
                # update mean
                # cp_list = [
                #     c_up_inv_xi_list[i] @ phi_xi_list_func(i, k_list=j_active_slice).T
                #     for i, xp in enumerate( x_up_list )
                # ]
                # optimized
                # pij_list = [
                #     # phi_xi_list[p][j_active_slice,:][:,stci_list[p]]
                #     phi_xi_list_batch[p][j_active_slice,:]
                #     # for p, xp in enumerate( x_up_list )
                #     for p in range( n_sample )
                # ]
                pij_list = phi_xi_list_batch_active
                cp_list = [
                    ( pij_list[p] * sigma**(-2) - pij_list[p] @ c_up_inv_xi_wmi_list[p][2].T @ c_up_inv_xi_wmi_list[p][3].T @ c_up_inv_xi_wmi_list[p][0].T ).T
                    # for p, xp in enumerate( x_up_list )
                    for p in range( n_sample )
                ]

                z_bar *= 0
                s_ = np.zeros(shape=( len(j_active_slice), cp_list[0].shape[1] ), dtype= cp_list[0].dtype)
                # for i, xp in enumerate( x_up_list ):
                for p in range( n_sample ):
                    # s_ += phi_xi_list[p][j_active_slice,:][:,stci_list[p]] @ cp_list[p]
                    s_ += pij_list[p] @ cp_list[p]
                np.fill_diagonal( s_, s_.diagonal() + 1 )
                # ycp = np.zeros(shape=( y_up_list[0][...,stci_list[0]].shape[0], cp_list[0].shape[1] ), dtype= cp_list[0].dtype)
                ycp = np.zeros(shape=( y_up_list_batch[0].shape[0], cp_list[0].shape[1] ), dtype= cp_list[0].dtype)
                # for i, xp in enumerate( x_up_list ):
                for p in range( n_sample ):
                    # ycp += y_up_list[p][...,stci_list[p]] @ cp_list[p]
                    ycp += y_up_list_batch[p] @ cp_list[p]
                z_bar[:, j_active_slice] = ycp @ svd_inverse(
                    s_,
                    cond_max= self.cond_num_max,
                    logging_prefix = f'[z_bar] ',
                    rcond_pinv = self.rcond_pinv,
                )[1]

                

                # y_up_list_centered = [
                #     yi-z_bar[:,j_active_slice] @ phi_xi_list[i][j_active_slice,:]
                #     for i, yi in enumerate( y_up_list)
                # ]
                y_up_list_centered_batch = [
                    # yi-z_bar[:,j_active_slice] @ phi_xi_list_batch[i][j_active_slice,:]
                    yi-z_bar[:,j_active_slice] @ phi_xi_list_batch_active[i]
                    for i, yi in enumerate( y_up_list_batch)
                ]



                if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
                    ln_like = self.get_ln_like( y_up_list = y_up_list_batch, z_bar=z_bar, beta = beta, sigma = sigma, phi_xi_list=phi_xi_list_batch ) if compute_loglike else -np.nan
                    logging.debug(f'mean [precision_update_fast] n_iter= {n_iter}, n_non_improve= {n_non_improve}, log likelihood= {ln_like :.8e}, sigma={sigma:.8e}, j_beta_active={ pprint.pformat( self.zip_beta( beta= beta), indent= 4) } \nz_bar[:, j_active_slice]={z_bar[:, j_active_slice]}')


            
            if dir_debug is not None:
                if n_iter % n_log == 0:
                    self.plot_results(
                        z_bar= z_bar,
                        beta= beta,
                        sigma= sigma,
                        path_fig= f'{dir_debug}/init_beta.iter{n_iter}.pdf')


            ln_like = self.get_ln_like( y_up_list = y_up_list_batch, z_bar=z_bar, beta = beta, sigma = sigma, phi_xi_list=phi_xi_list_batch ) if compute_loglike else -np.nan


            if ( compute_loglike and (ln_like - best_return['ln_like'] <= tol)):
                n_non_improve += 1
            else:
                n_non_improve = 0
            
            # HACK always act as if it is on plateau if compute_loglike is False to add more basis functions
            if ( (not compute_loglike) or (ln_like - best_return['ln_like'] <= tol_plateau) ):
                on_plateau= True
            else:
                on_plateau= False


            logging.info(f'[precision_update_fast] n_iter= {n_iter}, n_non_improve= {n_non_improve}, log likelihood= {ln_like :.8e}, sigma={sigma:.8e}, j_beta_active={ pprint.pformat( self.zip_beta( beta= beta), indent= 4) } \nz_bar[:, j_active_slice]={z_bar[:, j_active_slice]}')

            if (not compute_loglike) or (ln_like > best_return['ln_like']):
                best_return = {
                    'beta': beta.copy(), 
                    'z_bar': z_bar.copy(), 
                    # 'j2j_active': j2j_active.copy(),
                    'j_active_set': j_active_set.copy(),
                    'sigma': copy.deepcopy(sigma),
                    'ln_like': ln_like,
                }

            if n_non_improve >= n_patience:
                logging.info(f'[precision_update_fast] It has converged!')
                converged = True
            elif ( n_iter >= n_iter_max ):
                logging.warning(f'[precision_update_fast] optimization of beta did not converge, consider increasing the iterations!')
                break

        # returning
        z_bar, beta, j_active_set, sigma, ln_like = best_return['z_bar'], best_return['beta'], best_return['j_active_set'], best_return['sigma'], best_return['ln_like']
        logging.info(f'[precision_update_fast] n_iter= {n_iter}, returning, log likelihood= {ln_like :.8e}, sigma={sigma:.8e}, j_beta_active={ pprint.pformat( self.zip_beta( beta= beta ), indent= 4) }')
        logging.debug(f'j_list_deleted={j_list_deleted}, {len([j for j in j_active_set if j in j_list_deleted])} of j_active_set has been delected at least once')


        return z_bar, beta, sigma


    def plot_results(
        self,
        z_bar,        
        beta,
        sigma,

        path_fig = None,
        beta_threshold = None,

        phi_active_grid= None,

        x_up_d0_grid = None,
        d0_index_set_plot = 0,

        x_up_list= None,
        y_up_list= None,
        x_base_index= None,

        n_max_point_per_func_display= 1000,

        n_vis_limit= None,

        compute_loglike= True,
    ):
        '''
        x_up_d0_grid:
            (n_grid), grid index set for plotting the results
        phi_active_grid:
            (n_active, n_grid), active basis functions evaluated at x_up_d0_grid
        d0_index_set_plot:
            int, index of the dimension of index set to be plotted
        n_max_point_per_func_display:
            int, maximum number of points to be displayed for each functional data
        n_vis_limit:
            int, maximum number of functional data to be displayed, if None, will be determined by the default color map
        '''

        c_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
        n_vis_limit = len(c_list)-1 if n_vis_limit is None else n_vis_limit

        x_up_list= self.x_up_list if (x_up_list is None) else x_up_list
        y_up_list= self.y_up_list if (y_up_list is None) else y_up_list

        for i in range(len(y_up_list)):
            if y_up_list[i].shape[1] > n_max_point_per_func_display:
                logging.warning(f'[plot_results] y_up_list has more than {n_max_point_per_func_display} points for one function, only the first {n_max_point_per_func_display} points are displayed')
                break

        x_base_index= self.x_base_index if (x_base_index is None) else x_base_index

        x_up_list_mu_ni= np.array([(xi.mean(axis=1).flatten().tolist() + [xi.shape[1]]) for xi in x_up_list])
        x_up_list_mu = ( x_up_list_mu_ni[:, :-1] * x_up_list_mu_ni[:, [-1]] ).sum(axis=0) /  x_up_list_mu_ni[:, [-1]].sum()


        # NOTE only use the first dimension of index set for visualization
        x_up_list_d0 = [ xi[...,0] for xi in x_up_list ]
        x_base_index_d0 = x_base_index[..., d0_index_set_plot]


        beta_threshold = self.get_max_precision_active_dynamic_beta(beta) if (beta_threshold is None) else beta_threshold

        d_index_set = self.x_base_index.shape[1]

        if x_up_d0_grid is None:
            x_up_grid_max = np.array([ z.max() for z in x_up_list_d0]).max()
            x_up_grid_min = np.array([ z.min() for z in x_up_list_d0]).min()
            x_up_d0_grid = np.linspace(start=x_up_grid_min, stop=x_up_grid_max,)
        # (n_grid, d_index_set), grid index set for plotting the results, default values are the center of the index set
        x_up_grid = np.ones((x_up_d0_grid.shape[0], d_index_set))
        for d in range(d_index_set):
            if d == d0_index_set_plot:
                x_up_grid[:, d] = x_up_d0_grid[:]
            else:
                index_center_d = self.x_base_index[:, d].mean()
                x_up_grid[:, d] *= index_center_d


        x_minmax = (np.min(x_up_d0_grid), np.max(x_up_d0_grid))
        x_base_index_repeat = np.array(x_base_index_d0.tolist() * len(self.kernel_list))
        n_basis_func = len( x_base_index_repeat )
        n_sample = len( y_up_list )


        fig,ax = plt.subplots(6, 1, figsize = (16, 60))


        k2k_active_beta= {}
        max_precision_active_dynamic_beta_ = self.get_max_precision_active_dynamic_beta(beta)
        for j,aj in enumerate(beta):
            if aj < max_precision_active_dynamic_beta_:
                k2k_active_beta[j] = len(k2k_active_beta)
        k_active_beta = tuple( k2k_active_beta.keys() )


        # precompute
        x_i_e_list_active, _, _ = self.get_update_z_e(
            z_bar= z_bar,
            beta= beta,
            sigma= sigma,
            active_only= True,
        )


        ln_like= self.get_ln_like( y_up_list = y_up_list, z_bar=z_bar, beta = beta, sigma = sigma, phi_xi_list=self.phi_xi_list ) if compute_loglike else -np.nan

        k_active_beta = sorted(list(k2k_active_beta.keys()))
        phi_active_grid = ( self.phi( x_up_grid, x_base_index= x_base_index ) ) [ k_active_beta, : ] if phi_active_grid is None else phi_active_grid
        mu_active = np.concatenate( x_i_e_list_active, axis=0)

        a_up_n05_active = beta[ k_active_beta, None ].flatten()**(-0.5)
        z_bar_active = z_bar[:, k_active_beta]


        # original and fitted functional data
        y_fit = (z_bar_active+mu_active) @ phi_active_grid
        y_mean = (z_bar_active) @ phi_active_grid
        for i in range(min(n_vis_limit, len(y_up_list)),):
            ax[0].scatter(
                x= x_up_list_d0[i].flatten()[:n_max_point_per_func_display],
                y= y_up_list[i].flatten()[:n_max_point_per_func_display], 
                label=f'sample_{i}',
                s = ( plt.rcParams['lines.markersize'] ** 2 ) / 4,
                alpha= 0.9,
                )
            ax[1].scatter(
                x= x_up_list_d0[i].flatten()[:n_max_point_per_func_display], 
                y= y_up_list[i].flatten()[:n_max_point_per_func_display], 
                label=f'sample_{i}',
                s = ( plt.rcParams['lines.markersize'] ** 2 ) / 4,
                alpha= 0.9,
                )
            ax[1].plot( x_up_d0_grid, y_fit[i, :], label=f'sample_{i}_fit')
        ax[1].plot( x_up_d0_grid, y_mean[0], linestyle='--', label=f'mean_fit')


        # relevance vectors, estimation and truth
        rel_vec = [ (i,d) for i, d in enumerate( beta ) if d < beta_threshold] 
        for i, di in enumerate( rel_vec ):
            ax[0].axvline(x= x_base_index_repeat[di[0]], label= f'rel_vec_{i}@{x_base_index_repeat[di[0]]:.3e}' if i < n_vis_limit else '', c = 'g', ls= '-')
            ax[1].axvline(x= x_base_index_repeat[di[0]], label= f'rel_vec_{i}@{x_base_index_repeat[di[0]]:.3e}' if i < n_vis_limit else '', c = 'g', ls= '-')
            ax[3].axvline(x= x_base_index_repeat[di[0]], label= f'rel_vec_{i}@{x_base_index_repeat[di[0]]:.3e}' if i < n_vis_limit else '', c = 'g', ls= '-')


        # active kernel basis functions
        for i, (aj, j, jd) in enumerate( sorted( [ (beta[j], j, jd) for j,jd in k2k_active_beta.items() ] ) [: len(c_list)] ):
            ax[2].plot( x_up_d0_grid, phi_active_grid[jd, :], label=f'phi_active_{jd},beta={aj:.3e}@{x_base_index_repeat[j]:.3e}' if i < n_vis_limit else '', c = c_list[i % len(c_list)])
            ax[2].axvline(
                x= x_base_index_repeat[j], 
                ls= '-', color = c_list[i % len(c_list)],)


        # beta and weight
        ax4t = ax[3].twinx()
        m = np.zeros( shape= ( len( x_base_index_repeat ), len(y_up_list) ) )
        for i, (j, ja) in enumerate(k2k_active_beta.items()):
            m[j, : ] = mu_active.T[ ja, : ]
            # ax[3].axvline(x= x_base_index_repeat[ j ], label= f'beta_index_{ja}' if i < n_vis_limit else '', c = 'g', ls= '-')
        ax[3].scatter( 
            x = x_base_index_repeat, y = np.log10( beta ), label = 'lg(beta)', c = 'b',
            s = ( plt.rcParams['lines.markersize'] ** 2 ) / 4,
            alpha= 0.9,
            )
        for i in range(min(n_vis_limit, len(y_up_list)),):
            ax4t.scatter( 
                x = x_base_index_repeat, y = ( m[:, i] ), marker = 'x', label = f'weight_sample_{i}',
                s = ( plt.rcParams['lines.markersize'] ** 2 ) / 4,
                alpha= 0.9,
                )


        # spectrum analysis of the weight posterior
        ax5tx = ax[4].twinx()
        ax5txy = ax5tx.twiny()
        ax6tx = ax[5].twinx()
        ax6txy = ax6tx.twiny()

        phi_active_new = a_up_n05_active[:,None] * phi_active_grid
        u,s,v = np.linalg.svd( phi_active_new )
        stds = np.diag( phi_active_new @ phi_active_new.T )**0.5
        
        ax6txy.plot( s, label= 'std', marker='x' )
        components_stds = [
            (
                phi_active_new[ (i,), : ] / si * np.sign( phi_active_new[ (i,), np.argmax(np.abs(phi_active_new[ (i,), : ])) ]  ), 
                si
            )
            for i, si in enumerate( stds )
        ]
        components_stds = sorted( components_stds, key= lambda e:e[1], reverse= True)
        ax5txy.plot( [e[1] for e in components_stds], label= 'std', marker='x' )

        for i in range( min( len( components_stds ), n_vis_limit ) ):
            ui_phi, si_phi = components_stds[ i ]
            ax[4].plot( x_up_d0_grid, ui_phi.flatten(), label=f'component_{i}_(std={si_phi:.4e})')
            ax[5].plot( 
                x_up_d0_grid, 
                v[i].flatten() * np.sign( v[ (i,), np.argmax(np.abs(v[ (i,), : ])) ]  ), 
                label=f'component_{i}_(std={s[i]:.4e})')


        x_minmax_plot = (1.1 * x_minmax[0] - 0.1 * x_minmax[1],  1.1 * x_minmax[1] - 0.1 * x_minmax[0])
        ax[0].set_xlim(x_minmax_plot[0], x_minmax_plot[1])
        ax[1].set_xlim(x_minmax_plot[0], x_minmax_plot[1])
        ax[2].set_xlim(x_minmax_plot[0], x_minmax_plot[1])
        ax[3].set_xlim(x_minmax_plot[0], x_minmax_plot[1])
        ax4t.set_xlim(x_minmax_plot[0], x_minmax_plot[1])

        ax[0].set_title(f'n_basis_func={n_basis_func}, n_sample={n_sample}. center at [{",".join([f"{e:.3e}" for e in x_up_list_mu.tolist()])}]')
        ax[1].set_title(f'ln_like= {ln_like:.8e}, sigma_est={sigma:.8e}')

        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y_up_list')
        ax[0].legend()
        ax[0].grid()

        ax[1].set_xlabel('x')
        ax[1].set_ylabel('y')
        ax[1].legend()
        ax[1].grid()

        ax[2].set_xlabel('x')
        ax[2].set_ylabel(f'y')
        ax[2].legend()
        ax[2].grid()
        ax[2].set_title(f'active basis functions')

        ax[3].set_xlabel('x')
        ax[3].set_ylabel(f'lg(beta)')
        ax4t.set_ylabel(f'weight')
        ax[3].legend( loc='center left')
        ax4t.legend( loc='center right' )
        ax[3].grid()
        ax[3].set_title(f'beta and weight')

        ax[4].set_xlabel('x')
        ax[4].set_ylabel(f'y')
        ax[4].legend(loc='upper right')
        ax[4].grid()
        ax[4].set_title(f'normalized phi_active_new ( diag(β^(-0.5)) Φ )')

        ax6txy.set_xlabel('i_std')
        ax6tx.set_ylabel('std')
        ax6txy.legend( loc= 'upper center')

        ax5txy.set_xlabel('i_std')
        ax5tx.set_ylabel('std')
        ax5txy.legend( loc= 'upper center')

        ax[5].set_xlabel('x')
        ax[5].set_ylabel(f'y')
        ax[5].legend(loc='upper right')
        ax[5].grid()
        ax[5].set_title(f'SVD of phi_active_new ( diag(β^(-0.5)) Φ )')


        plt.tight_layout()
        
        if path_fig is not None:
            fig.savefig(path_fig)
        
        plt.close( fig )

        return path_fig



class nRVM_v1():
    '''
    yi = ( z_bar + zi ) @ phi_i + ei, zi ~ N(0, diag(beta**(-1))), ei ~ N(0, sigma**2 * I )
    '''
    def __init__(
        self,
        kernel_list,
        max_precision_active = EPS**(-1),
        cond_num_max = None,
        ) -> None:
        '''
        max_precision_active:
            maximum allowed precision of the active basis functions, for numerical stability
        cond_num_max:
            maximum allowed condition number of the matrix for solving z_bar, for numerical stability
        '''
        self.cond_num_max= cond_num_max
        self.kernel_list = kernel_list
        self.max_precision_active = max_precision_active
        self.j_list_similar2active = []
        pass

    def phi(
        self, 
        x, 
        x_base_index,
        ):
        '''
        calculate the matrix of basis functions
        ---
        x:
            (n_point, d_index)
        x_base_index:
            (n_base_index, d_index)
        ---
        return:
            (n_base_index * n_kernel, n_point,)
        '''
        return np.concatenate([
            k(x_base_index, x)
            for k in self.kernel_list
        ], axis=0)



    def kms_cluster4candidate_bases(
        self,
        x_up_list,
        n_x_base_index = None,
        coef_func_index= None,
    ):
        '''
        get the basis functions at the candidates of relevacne vectors using kmeans++ clustering
        ---
        x_up_list:
            a list of n_sample (1, n_up_i, d_index) arrays
        coef_func_index:
            multiplier for the number of relevance vector candidates
        n_x_base_index:
            number of candidates basis functions
        ---
        return:
            x_base_index:
                (n_x_base_index, d_index) ndarray
        '''
        x_up_all = []
        for x in x_up_list:
            x_up_all = x_up_all + x[0].tolist()
        # (n_up_all, d_index)
        x_up_all = np.array( x_up_all )


        if n_x_base_index is None:

            if coef_func_index is not None:
                n_x_base_index = int(np.max([
                    xi.shape[1] for xi in x_up_list
                ]) * coef_func_index)

            else:
                
                # set the number of index set candidates for the relevance vectors
                # the maximum number of index set candidates is 1e4, which is the total number of relevance vectors in the 4d simulation and works weel
                n_x_base_index = min( len(x_up_all), int(1e4) )
        

        if n_x_base_index < len(x_up_all):

            km = KMeans(
                n_clusters= n_x_base_index,
                init= 'k-means++',
            )

            km.fit(X= x_up_all)

            # (n_cluster, d_index)
            centers = km.cluster_centers_

            # (n_cluster, d_index) base index that is closest to the center of each cluster
            x_base_index = x_up_all[ np.argmin( ( ( centers[:,None,:] - x_up_all[None,:,:] )**2 ).sum(axis=2) , axis=1 ), : ]


        else:

            x_base_index = x_up_all


        # sort from the first dimension to the last dimension of the index set ascendingly
        x_base_index_sorted = x_base_index[ np.lexsort(x_base_index[:,::-1].T) ]


        return x_base_index_sorted



    def init_beta(
        self,
        phi_xi_list_func,
    ):
        '''
        # x_base_index:
        #     (n_base, d_index) ndarray of the candidates of index set of the relevance vectors
        ---
        use the given initial parameters or use a heuristic algorithm to initialize the parameters
        '''
        x_up_list= self.x_up_list.copy()
        y_up_list= self.y_up_list.copy()
        n_base= self.n_base


        beta = np.ones( shape= (n_base) ) * self.max_precision_active

        # add the most likely relevance vector
        k_init_beta = np.argmax( np.sum(
            [
                ( y_up_list[ i ] @ (phi_xi_list_func(i) / (phi_xi_list_func(i)**2).sum(axis=1,keepdims=True)**0.5).T )[0]**2
                for i, xi in enumerate( x_up_list )
            ],
            axis=0
        ) )

        logging.info(f'[param_init] k_init_beta={k_init_beta}')

        beta[k_init_beta] = 1


        return beta


    def get_ln_like(
        self,
        y_up_list,

        z_bar,
        beta,
        sigma,
        phi_xi_list_func,

        b_up = None,
        d_low = None,
        k2k_active_d_low= None,
        ):
        '''
        likelihood of Yi = Xi @ diag(a)**(-0.5) @ B @ diag(d)**(-0.5) @ Phii + sigma * ei, Xi and ei obey the standard muti variate normal distribution
        ---
        y_up_list:
            a list of n_sample (1, n_index_i, ) arrays
        beta:
            precisions, (1, n_bases)
        b_up:
            identity when None
        d_low:
            identity when None
        sigma:
            standard deviation of white noise
        phi_xi_list_func:
            Φ
        '''

        k2k_active_beta= {}
        max_precision_active_dynamic_beta_ = self.get_max_precision_active_dynamic_beta(beta)
        for j,aj in enumerate(beta):
            if aj < max_precision_active_dynamic_beta_:
                k2k_active_beta[j] = len(k2k_active_beta)

        # a_n1 = np.diag( beta.flatten() ** (-1) )
        k_active_beta = sorted(list( k2k_active_beta.keys() ))

        if d_low is None:
            if b_up is None:
                bdf_list = [
                    phi_xi_list_func(i, k_list= k_active_beta)
                    for i, yi in enumerate( y_up_list)
                ]
            else:
                bdf_list = [
                    b_up[ k_active_beta, : ] @ phi_xi_list_func(i)
                    for i, yi in enumerate( y_up_list)
                ]
        else:
            j_active_d_low = sorted(list( k2k_active_d_low.keys() ))
            d_low_n05_active = np.diag( d_low[ j_active_d_low ] ** (-0.5) )
            
            if b_up is None:
                bdf_list = [
                    d_low_n05_active @ phi_xi_list_func(i, k_list=j_active_d_low)
                    for i, yi in enumerate( y_up_list)
                ]
            else:
                bdf_list = [
                    b_up[ np.ix_( k_active_beta, j_active_d_low, ) ] @ d_low_n05_active @ phi_xi_list_func(i, k_list=j_active_d_low)
                    for i, yi in enumerate( y_up_list)
                ]

        y_up_list_centered = [
            yi-z_bar[:,k_active_beta] @ phi_xi_list_func(i, k_list=k_active_beta)
            for i, yi in enumerate( y_up_list)
        ]

        '''
        # NOTE can be parallelized
        # matrix inverse lemma
        cov_inv_list = [
            woodbury_inverse(
                a_inv = sigma**(-2) * np.identity( n= bdf_list[i].shape[1] ),
                u= bdf_list[i].T,
                c_inv = np.diag( beta[ k_active_beta ] ),
                v= bdf_list[i],
            )
            for i, yi in enumerate( y_up_list)
        ]

        # NOTE can be parallelized
        l_old = np.sum([
            - 1/2 * (
                ( yi.shape[1] ) * np.log( 2 * np.pi) - np.linalg.slogdet( ( cov_inv_list[i] ) )[1] + y_up_list_centered[i] @ cov_inv_list[i] @ y_up_list_centered[i].T
            ) for i, yi in enumerate( y_up_list)
        ])
        '''


        cov_inv_wmi_list = [
            woodbury_inverse(
                a_inv = sigma**(-2) * np.identity( n= bdf_list[i].shape[1] ),
                u= bdf_list[i].T,
                c_inv = np.diag( beta[ k_active_beta ] ),
                v= bdf_list[i],
                return_parts= True,
            )
            for i, yi in enumerate( y_up_list)
        ]
        # cov_inv = a_inv - wmi[0] @ inv(wmi[1]) @ wmi[2]

        # matrix determinant lemma
        cov_inv_logdet_list = [
            woodbury_logdet(
                a_e = sigma**(-2),
                u = cov_inv_wmi_list[i][0],
                w_inv = cov_inv_wmi_list[i][1],
                vt = cov_inv_wmi_list[i][2],
            )
            for i, yi in enumerate( y_up_list)
        ]


        l = np.sum([
            - 1/2 * (
                ( yi.shape[1] ) * np.log( 2 * np.pi) - cov_inv_logdet_list[i] + ( y_up_list_centered[i] * sigma**(-2) @ y_up_list_centered[i].T - y_up_list_centered[i] @ cov_inv_wmi_list[i][0] @ np.linalg.inv( cov_inv_wmi_list[i][1] ) @ cov_inv_wmi_list[i][2] @ y_up_list_centered[i].T )
            ) for i, yi in enumerate( y_up_list)
        ])

        return l


    def get_update_z_e(
        self,
        z_bar,
        beta,
        sigma,
        eps= EPS,
        active_only=False,
        ):
        '''
        calculate the poterior of the latent variables
        '''


        y_up_list= self.y_up_list.copy()
        phi_phit_xi_list_func = self.phi_phit_xi_list_func


        k2k_active_beta= {}
        max_precision_active_dynamic_beta_ = self.get_max_precision_active_dynamic_beta(beta)
        for j,aj in enumerate(beta):
            if aj < max_precision_active_dynamic_beta_:
                k2k_active_beta[j] = len(k2k_active_beta)

        k_active_beta = sorted(list(k2k_active_beta.keys()))

        
        n_index_bases = len(self.x_base_index)
        phi_xi_list_active = [
            np.concatenate(
                [
                    self.phi( xi[0,:], x_base_index= self.x_base_index[ [j % n_index_bases] ] ) [ [j // n_index_bases], : ]
                    for j in k_active_beta
                ], axis=0
            ) 
            for xi in self.x_up_list
        ]
        y_phit_xi_list_active = [
            y_i @ phi_xi_list_active[i].T
            for i, y_i in enumerate( y_up_list )
        ]


        a_active = np.diag( beta[ k_active_beta, ].flatten() )


        phi_phit_xi_list_active = [
            phi_phit_xi_list_func(i=i, j1_list= k_active_beta, j2_list= k_active_beta)
            for i, y_i in enumerate( y_up_list )
        ]

        mi_list_active = [
            svd_inverse(
                sigma**(-2) * phi_phit_xi_list_active[i] + a_active,
                jitter= eps**0.5,
                logging_prefix = f'[mi_list_active] ',
            ) 
            for i, y_i in enumerate( y_up_list )
        ]

        z_i_e_list_active = [
            sigma**(-2) * ( y_phit_xi_list_active[i] - z_bar[:,k_active_beta] @ phi_phit_xi_list_active[i] ) @ mi_list_active[i] 
            for i, y_i in enumerate( y_up_list )
        ]

        ztz_i_e_list_active = [
            mi_list_active[i] + z_i_e_list_active[i].T @ z_i_e_list_active[i]
            for i, y_i in enumerate( y_up_list )
        ]

        if not active_only:
            ztz_i_e_list = [
                np.identity( n= beta.shape[0] )
                for i, y_i in enumerate( y_up_list )
            ]
            z_i_e_list = [
                np.zeros( shape= (1, beta.shape[0]) )
                for i, y_i in enumerate( y_up_list )
            ]
            for i, y_i in enumerate( y_up_list ):
                z_i_e_list[i][ 0, k_active_beta ] = z_i_e_list_active[i][ 0, : ]

                ztz_i_e_list[i][ np.ix_( k_active_beta, k_active_beta ) ] = ztz_i_e_list_active[i]


            return z_i_e_list, ztz_i_e_list
        else:
            return z_i_e_list_active, ztz_i_e_list_active
            


    def init_params(
        self,
        coef_func_index,
        x_base_index,
        ratio_init_var_noise,
        sigma_init,
    ):
        x_up_list= self.x_up_list
        y_up_list= self.y_up_list

        
        # (n_x_base_index, d_index)
        self.x_base_index = self.kms_cluster4candidate_bases(
            coef_func_index=coef_func_index,
            x_up_list= x_up_list,
        ) if ( x_base_index is None) else x_base_index

        logging.info(f'x_base_index.shape= {self.x_base_index.shape}, self.x_base_index= {pprint.pformat( self.x_base_index, indent= 4)}')

        phi_xi_list = [
            self.phi( xi[0,:], x_base_index= self.x_base_index ) 
            for xi in x_up_list
        ]

        def phi_xi_list_func(
            i,
            k_list='all',
            phi_xi_list = phi_xi_list,
        ):
            if k_list=='all':
                return phi_xi_list[i]
            else:
                return phi_xi_list[i][ k_list, : ]

        # list of n_sample (n_base, n_index_i) arrays
        self.phi_xi_list_func = phi_xi_list_func
        self.n_base = phi_xi_list_func(0).shape[0]

        # mean of the latent variables
        self.z_bar = np.zeros(shape=(1, self.n_base))


        # precompute
        self.phi_phit_xi_list_func = lambda i, j1_list, j2_list, phi_xi_list_func= phi_xi_list_func : phi_xi_list_func(i, k_list=j1_list) @ phi_xi_list_func(i, k_list=j2_list).T

        # initialize beta
        self.beta = self.init_beta(
            phi_xi_list_func= phi_xi_list_func,
        )

        self.k2k_active_beta = OrderedDict()
        max_precision_active_dynamic_beta_ = self.get_max_precision_active_dynamic_beta(self.beta)
        for i, ei in enumerate( np.argwhere( self.beta < max_precision_active_dynamic_beta_ ) ):
            self.k2k_active_beta[ int(ei) ] = i

        logging.info(f'[param_init] j_beta_active={ pprint.pformat( self.zip_beta( beta= self.beta, ), indent= 4) }')


        # assign a sensable variance to the white noise
        if (sigma_init is None):

            self.sigma = ( ratio_init_var_noise * np.sum([ yp@yp.T for yp in y_up_list]) / np.sum([ len(yp) for yp in y_up_list ]) )**0.5

            logging.info(f'[param_init] sigma={self.sigma :.8e}, log likelihood= {self.get_ln_like( y_up_list = y_up_list, z_bar=self.z_bar, beta = self.beta, sigma = self.sigma, phi_xi_list_func=self.phi_xi_list_func ) :.8e}')

            # initialize sigma using em
            logging.info(f'initialize sigma')
            z_i_e_list_active, ztz_i_e_list_active = self.get_update_z_e(
                z_bar = self.z_bar,
                beta= self.beta,
                sigma= self.sigma,
                active_only= True,
            )

            self.sigma = self.get_mstep_update_sigma(
                z_bar = self.z_bar,
                beta= self.beta,
            
                z_i_e_list_active= z_i_e_list_active,
                ztz_i_e_list_active= ztz_i_e_list_active,
            )
            logging.info(f'[param_update] sigma={self.sigma :.8e}, log likelihood= {self.get_ln_like( y_up_list = y_up_list, z_bar=self.z_bar, beta = self.beta, sigma = self.sigma, phi_xi_list_func=self.phi_xi_list_func ) :.8e}')

        else:
            self.sigma = sigma_init
    
            logging.info(f'[param_init] sigma={self.sigma :.8e}, log likelihood= {self.get_ln_like( y_up_list = y_up_list, z_bar=self.z_bar, beta = self.beta, sigma = self.sigma, phi_xi_list_func=self.phi_xi_list_func ) :.8e}')





        pass

    def get_max_precision_active_dynamic_beta(
        self,
        beta,
        ):
        '''
        this currently does nothing but return self.max_precision_active
        '''
        return self.max_precision_active


    def get_mstep_update_sigma(
        self,
        z_bar,
        beta,

        z_i_e_list_active,
        ztz_i_e_list_active,
        ):
        '''
        m step for sigma
        '''
        y_up_list= self.y_up_list.copy()

        phi_phit_xi_list_func = self.phi_phit_xi_list_func

        k2k_active_beta= {}
        max_precision_active_dynamic_beta_ = self.get_max_precision_active_dynamic_beta(beta)
        for j,aj in enumerate(beta):
            if aj < max_precision_active_dynamic_beta_:
                k2k_active_beta[j] = len(k2k_active_beta)
        k_active_beta = sorted(tuple(k2k_active_beta.keys()))


        phi_phit_xi_list_active = [
            phi_phit_xi_list_func(i=i, j1_list= k_active_beta, j2_list= k_active_beta)
            for i, yi in enumerate( y_up_list )
        ]

        y_phit_list_centered_active = [
            yi - (z_bar[:, k_active_beta] @ self.phi_xi_list_func(i, k_list= k_active_beta))
            for i, yi in enumerate( y_up_list )
        ]
        

        sigma_new_sq = np.sum([
            yi.shape[1]
            for i, yi in enumerate( y_up_list )
        ]).astype(np.float)**(-1) * \
            np.sum([
                y_phit_list_centered_active[i] @ y_phit_list_centered_active[i].T - 2 * y_phit_list_centered_active[i] @ self.phi_xi_list_func(i, k_list= k_active_beta).T @ z_i_e_list_active[i].T + \
                    np.trace( ztz_i_e_list_active[i] @ phi_phit_xi_list_active[i] )
                for i, yi in enumerate( y_up_list )
            ])
        
        sigma_new = sigma_new_sq**(0.5)

        return sigma_new


    def zip_beta(
        self,
        beta,
    ):
        x_base_index_all_kernel = np.array( self.x_base_index.tolist() * len(self.kernel_list) )

        k2k_active_beta= {}
        max_precision_active_dynamic_beta_ = self.get_max_precision_active_dynamic_beta(beta)
        for j,aj in enumerate(beta):
            if aj < max_precision_active_dynamic_beta_:
                k2k_active_beta[j] = len(k2k_active_beta)

        jas= sorted(list( k2k_active_beta.keys() ))
        j_beta_x_active = sorted(
            zip(
                jas,
                beta[ jas ],
                x_base_index_all_kernel[ jas ]
            ),
            key= lambda e:e[1],
        )
        return j_beta_x_active


    def fit(
        self,
        x_up_list,
        y_up_list,
        # initialization
        coef_func_index= None,
        x_base_index=None,
        ratio_init_var_noise=0.1,
        sigma_init=None,
        # fast fitting
        n_iter_max_fast=100,
        n_patience= 10,
        tol= None,
        sigma_is_fixed= False,
        sim_threshold = 0.999,
        # em update
        tol_plateau= None,
        # trim
        n_log= 1,
        dir_debug= None,        
    ):
        self.x_up_list= x_up_list
        self.y_up_list= y_up_list

        tol_plateau = 1e-2 * len(y_up_list) if tol_plateau is None else tol_plateau

        self.init_params(
            coef_func_index= coef_func_index,
            x_base_index= x_base_index,
            ratio_init_var_noise= ratio_init_var_noise,
            sigma_init= sigma_init,
        )
        self.z_bar, self.beta, self.sigma = self.get_update_fast(
            z_bar= self.z_bar,
            beta= self.beta,
            sigma= self.sigma,

            n_iter_max= n_iter_max_fast,
            n_patience= n_patience,
            tol= tol,
            tol_plateau= tol_plateau,
            sigma_is_fixed= sigma_is_fixed,
            dir_debug= dir_debug if (dir_debug is None) else f'{dir_debug}/fast' ,
            
            n_log= n_log,
            sim_threshold = sim_threshold,
        )

        self.plot_results(
            z_bar= self.z_bar,
            beta= self.beta,
            sigma= self.sigma,

            path_fig= f'{dir_debug}/final.pdf')
        # self.dump( path_dump= f'{dir_debug}/final.pkl')


        logging.critical(f'nRVM fitting is done!')

        return self

    def dump(self, path_dump):
        '''
        use dill to dump the object, but it can be loaded by pickle
        '''
        with open(path_dump, 'wb') as f:
            dill.dump(
                obj= self,
                file= f,
            )

    
    def k_list_get_fm(
        self,
        j_list,
        ):
        n_index_bases = len(self.x_base_index)

        f_j_list = [
            RadialBasisFunction(
                length_scale= self.kernel_list[ j // n_index_bases ].length_scale,
                center= self.x_base_index[ [j % n_index_bases] ],
                multiplier= self.kernel_list[ j // n_index_bases ].multiplier,
                )
            for j in j_list
        ]

        fm_j = RadicalBasisFunctionMixture( radial_basis_function_list= f_j_list )

        return fm_j


    def get_update_fast(
        self,
        z_bar,
        beta,
        sigma,

        n_iter_max=100,
        n_patience= 10,
        tol= None,
        tol_plateau= None,
        sigma_is_fixed= False,

        sim_threshold = 0.999,
        n_log= 1,

        dir_debug= None,
        ):
        '''
        compute updates of precisions using direct differentiation with approximations
        ---
        n_iter_max:
            a positive integer
        '''

        tol= 1e-5 * len(x_up_list) if (tol is None) else tol
        tol_plateau= 1e-2 * len(self.x_up_list) if (tol_plateau is None) else tol_plateau


        x_up_list= self.x_up_list.copy()
        y_up_list= self.y_up_list.copy()


        # j2j_active= {}
        j_active_set= set()
        max_precision_active_dynamic_beta_ = self.get_max_precision_active_dynamic_beta(beta)
        for j,aj in enumerate(beta):
            if aj < max_precision_active_dynamic_beta_:
                # j2j_active[j] = len(j2j_active)
                j_active_set.add(j)
        j_active_slice = sorted(list(j_active_set))


        y_up_list_centered = [
            yi-z_bar[:,j_active_slice] @ self.phi_xi_list_func(i, k_list= j_active_slice)
            for i, yi in enumerate( y_up_list)
        ]


        if (dir_debug is not None) and (not os.path.exists( dir_debug )):
            os.makedirs( dir_debug )

        phi_xi_list_func = self.phi_xi_list_func
        n_base = phi_xi_list_func(0).shape[0]
        n_sample = len( x_up_list )


        s_up = np.zeros(shape= (n_base, n_sample))
        q_up = np.zeros(shape= (n_base, n_sample))
        theta = np.zeros_like( beta )

        # if beta changes, the covariances needs to be updated
        # j_active_slice = sorted(list(j2j_active.keys()))
        c_up_inv_xi_list = [
            np.linalg.inv( sigma**2 * np.identity(n= xp.shape[1]) + ( phi_xi_list_func(p, k_list=j_active_slice) ).T @ np.diag( beta[ j_active_slice ] ** (-1) ) @ ( phi_xi_list_func(p, k_list=j_active_slice) ) )
            for p, xp in enumerate( x_up_list )
        ]


        # here beta is directly next to basis functions thus it like d_low
        logging.info(f'[precision_update_fast] log likelihood= {self.get_ln_like( y_up_list = y_up_list, z_bar=z_bar, beta = beta, sigma = sigma, phi_xi_list_func=self.phi_xi_list_func ):.8e}, j_beta_active={ pprint.pformat( self.zip_beta( beta= beta ), indent= 4) }')


        converged = False
        on_plateau= False
        n_iter = 0
        ln_like = self.get_ln_like( y_up_list = y_up_list, z_bar=z_bar, beta = beta, sigma = sigma, phi_xi_list_func=self.phi_xi_list_func )


        n_non_improve = 0
        best_return = {
            'beta': beta.copy(), 
            'z_bar': z_bar.copy(), 
            # 'j2j_active': j2j_active.copy(),
            'j_active_set': j_active_set.copy(),
            'sigma': copy.deepcopy(sigma),
            'ln_like': ln_like.copy(),
        }
        # history of deletion of the basis functions
        j_list_deleted = []
        # NOTE self.j_list_similar2active need to be updated when there is an active basis functions deleted because the similarity could decrease
        self.j_list_similar2active = []

        while ( not converged ):

            n_iter += 1

            j_list = list( range( len( beta )) )
            np.random.shuffle( j_list )


            # order the basis functions based on theta and similarity
            if on_plateau:

                j2sim2active = {}

                # fm_active = self.k_list_get_fm( j_list= list( j2j_active.keys() ) )
                fm_active = self.k_list_get_fm( j_list= sorted(list( j_active_set )) )


                for j in j_list:
                    
                    # skip the basis functions that are already similar to the active ones
                    # if ( (j in self.j_list_similar2active) or (j in j2j_active) ):
                    if ( (j in self.j_list_similar2active) or (j in j_active_set) ):
                        continue

                    # cosine similarity between the new basis function and the optimized ones
                    # reuse fm_active to save time from orthornomalization
                    sim2active = fm_active.cosine_similar2subspace(
                        rbfm= self.k_list_get_fm( j_list= [j,] ),
                    )
                    logging.debug(f'j={j}, sim2active={sim2active}')


                    if sim2active < sim_threshold:

                        j2sim2active[j] = sim2active
                        q_up_j = np.array([ ( phi_xi_list_func(p, k_list=[j]) @ c_up_inv_xi_list[ p ] @ y_up_list_centered[ p ].T )[0,0] for p in range( n_sample )])
                        s_up_j = np.array([ ( phi_xi_list_func(p, k_list=[j]) @ c_up_inv_xi_list[ p ] @ phi_xi_list_func(p, k_list=[j]).T )[0,0] for p in range( n_sample )])
                        if beta[j] >= self.get_max_precision_active_dynamic_beta(beta):
                            q = q_up_j
                            s = s_up_j
                        else:
                            q = beta[j] * q_up_j / ( beta[j] - s_up_j )
                            s = beta[j] * s_up_j / ( beta[j] - s_up_j )
                        theta[j] = np.sum( q**2 - s )
                    
                    else:
                        self.j_list_similar2active.append(j)


                theta_j_candidate_list = [
                    (theta[j],j) for j in j2sim2active
                ]

                if len(theta_j_candidate_list)>0:
                    theta_j_candidate_list = sorted(theta_j_candidate_list, reverse=True)
                    j_candidate_list = np.array(theta_j_candidate_list).astype(int)[:,1].tolist()
                    logging.debug(f'[sorting new bases] theta_j_candidate_list={pprint.pformat(theta_j_candidate_list)}')
                    j = j_candidate_list[0]

                    logging.debug(f'consider the basis, j={j}, theta[j]={theta[j]}')

                    # add this basis function
                    if theta[j] > 0:
                        beta[j] = np.sum( s**2 ) / theta[j]

                        logging.debug(f'add the basis, beta[j]={beta[j]}')

                        # j2j_active[j] = len(j2j_active)
                        j_active_set.add(j)

                        on_plateau = False

            # j_list_active = list(j2j_active)
            j_list_active = list(j_active_set)
            np.random.shuffle(j_list_active)

            # update the active dimensions
            for j in j_list_active:
                q_up_j = np.array([ ( phi_xi_list_func(p, k_list=[j]) @ c_up_inv_xi_list[ p ] @ y_up_list_centered[ p ].T )[0,0] for p in range( n_sample )])

                s_up_j = np.array([ ( phi_xi_list_func(p, k_list=[j]) @ c_up_inv_xi_list[ p ] @ phi_xi_list_func(p, k_list=[j]).T )[0,0] for p in range( n_sample )])
                max_precision_active_dynamic_beta_ = self.get_max_precision_active_dynamic_beta(beta)
                if beta[j] >= max_precision_active_dynamic_beta_:
                    q = q_up_j
                    s = s_up_j
                else:
                    q = beta[j] * q_up_j / ( beta[j] - s_up_j )
                    s = beta[j] * s_up_j / ( beta[j] - s_up_j )
                theta[j] = np.sum( q**2 - s )


                if theta[j] > 0:

                    max_precision_active_dynamic_beta_ = self.get_max_precision_active_dynamic_beta(beta)
                    if beta[j] < max_precision_active_dynamic_beta_:

                        beta[j] = np.sum( s**2 ) / theta[j]

                        logging.debug(f're estimated beta[j]={beta[j]}')

                else:
                    
                    logging.debug('theta[j] <= 0')
                    
                    max_precision_active_dynamic_beta_ = self.get_max_precision_active_dynamic_beta(beta)
                    if beta[j] < max_precision_active_dynamic_beta_:

                        # if len( j2j_active ) > 1:
                        if len( j_active_set ) > 1:

                            logging.debug('delete the basis')
                            j_list_deleted.append(j)

                            beta[j] = max_precision_active_dynamic_beta_

                            # j_active = j2j_active[j]

                            # # update the indices that are active after deleting the current one
                            # j2j_active.pop(j)
                            # for k,v in j2j_active.items():
                            #     if v > j_active:
                            #         j2j_active[k] = v-1
                            j_active_set.remove(j)
                        
                            self.j_list_similar2active = []

                        else:
                            pass
                            logging.debug('skip deleting the basis, because this is the only basis')



                # j_active_slice = list(j2j_active.keys())
                j_active_slice = sorted(list(j_active_set))


                ln_like = self.get_ln_like( y_up_list = y_up_list, z_bar=z_bar, beta = beta, sigma = sigma, phi_xi_list_func=self.phi_xi_list_func )
                logging.debug(f'[precision_update_fast] n_iter= {n_iter}, n_non_improve= {n_non_improve}, log likelihood= {ln_like :.8e}, sigma={sigma:.8e}, j_beta_active={ pprint.pformat( self.zip_beta( beta= beta), indent= 4) } \nz_bar[:, j_active_slice]={z_bar[:, j_active_slice]}')

                # update white noise
                z_i_e_list_active, ztz_i_e_list_active = self.get_update_z_e(
                    beta= beta,
                    sigma= sigma,
                    z_bar = z_bar,
                    active_only= True,
                )
                if not sigma_is_fixed:
                    sigma = self.get_mstep_update_sigma(
                        beta= beta,
                        z_bar = z_bar,
                        z_i_e_list_active= z_i_e_list_active,
                        ztz_i_e_list_active= ztz_i_e_list_active,
                    )
                ln_like = self.get_ln_like( y_up_list = y_up_list, z_bar=z_bar, beta = beta, sigma = sigma, phi_xi_list_func=self.phi_xi_list_func )
                logging.debug(f'[precision_update_fast] n_iter= {n_iter}, n_non_improve= {n_non_improve}, log likelihood= {ln_like :.8e}, sigma={sigma:.8e}, j_beta_active={ pprint.pformat( self.zip_beta( beta= beta), indent= 4) } \nz_bar[:, j_active_slice]={z_bar[:, j_active_slice]}')


                # NOTE can be parallelized
                # matrix inverse lemma
                # update s_up and q_up
                c_up_inv_xi_list = [
                    woodbury_inverse(
                        a_inv= sigma**(-2) * np.identity(n= xp.shape[1]),
                        u= phi_xi_list_func(p, k_list=j_active_slice).T,
                        c_inv= np.diag( beta[ j_active_slice ] ),
                        v= phi_xi_list_func(p, k_list=j_active_slice)
                    )
                    for p, xp in enumerate( x_up_list )
                ]
                
                # NOTE z_bar is reset to zero and then updated
                # update mean
                cp_list = [
                    c_up_inv_xi_list[i] @ phi_xi_list_func(i, k_list=j_active_slice).T
                    for i, xp in enumerate( x_up_list )
                ]
                z_bar *= 0
                z_bar[:, j_active_slice] = np.sum([
                    y_up_list[i] @ cp_list[i]
                    for i, xp in enumerate( x_up_list )
                ], axis=0) @ svd_inverse(
                    np.sum([
                        phi_xi_list_func(i, k_list=j_active_slice) @ cp_list[i]
                        for i, xp in enumerate( x_up_list )
                    ],axis=0) + np.identity(len(j_active_slice)),
                    cond_max= self.cond_num_max,
                    logging_prefix = f'[z_bar] ',
                )

                

                y_up_list_centered = [
                    yi-z_bar[:,j_active_slice] @ phi_xi_list_func(i, k_list=j_active_slice)
                    for i, yi in enumerate( y_up_list)
                ]



                ln_like = self.get_ln_like( y_up_list = y_up_list, z_bar=z_bar, beta = beta, sigma = sigma, phi_xi_list_func=self.phi_xi_list_func )
                logging.debug(f'mean [precision_update_fast] n_iter= {n_iter}, n_non_improve= {n_non_improve}, log likelihood= {ln_like :.8e}, sigma={sigma:.8e}, j_beta_active={ pprint.pformat( self.zip_beta( beta= beta), indent= 4) } \nz_bar[:, j_active_slice]={z_bar[:, j_active_slice]}')



            if dir_debug is not None:
                if n_iter % n_log == 0:
                    self.plot_results(
                        z_bar= z_bar,
                        beta= beta,
                        sigma= sigma,
                        path_fig= f'{dir_debug}/init_beta.iter{n_iter}.pdf')



            ln_like = self.get_ln_like( y_up_list = y_up_list, z_bar=z_bar, beta = beta, sigma = sigma, phi_xi_list_func=self.phi_xi_list_func )


            if ln_like - best_return['ln_like'] <= tol:
                n_non_improve += 1
            else:
                n_non_improve = 0

            if ln_like - best_return['ln_like'] <= tol_plateau:
                on_plateau= True
            else:
                on_plateau= False


            logging.info(f'[precision_update_fast] n_iter= {n_iter}, n_non_improve= {n_non_improve}, log likelihood= {ln_like :.8e}, sigma={sigma:.8e}, j_beta_active={ pprint.pformat( self.zip_beta( beta= beta), indent= 4) } \nz_bar[:, j_active_slice]={z_bar[:, j_active_slice]}')

            if ln_like > best_return['ln_like']:
                best_return = {
                    'beta': beta.copy(), 
                    'z_bar': z_bar.copy(), 
                    # 'j2j_active': j2j_active.copy(),
                    'j_active_set': j_active_set.copy(),
                    'sigma': copy.deepcopy(sigma),
                    'ln_like': ln_like.copy(),
                }

            if n_non_improve >= n_patience:
                logging.info(f'[precision_update_fast] It has converged!')
                converged = True
            elif ( n_iter >= n_iter_max ):
                logging.warning(f'[precision_update_fast] optimization of beta did not converge, consider increasing the iterations!')
                break

        # returning
        z_bar, beta, j_active_set, sigma, ln_like = best_return['z_bar'], best_return['beta'], best_return['j_active_set'], best_return['sigma'], best_return['ln_like']
        logging.info(f'[precision_update_fast] n_iter= {n_iter}, returning, log likelihood= {ln_like :.8e}, sigma={sigma:.8e}, j_beta_active={ pprint.pformat( self.zip_beta( beta= beta ), indent= 4) }')
        logging.debug(f'j_list_deleted={j_list_deleted}, {len([j for j in j_active_set if j in j_list_deleted])} of j_active_set has been delected at least once')


        return z_bar, beta, sigma


    def plot_results(
        self,
        z_bar,        
        beta,
        sigma,

        path_fig = None,
        beta_threshold = None,

        phi_active_grid= None,

        x_up_d0_grid = None,
        d0_index_set_plot = 0,

        x_up_list= None,
        y_up_list= None,
        x_base_index= None,
        n_vis_limit= None,
    ):
        '''
        x_up_d0_grid:
            (n_grid), grid index set for plotting the results
        phi_active_grid:
            (n_active, n_grid), active basis functions evaluated at x_up_d0_grid
        d0_index_set_plot:
            int, index of the dimension of index set to be plotted
        n_vis_limit:
            int, maximum number of functional data to be displayed, if None, will be determined by the default color map
        '''

        c_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
        x_up_list= self.x_up_list if (x_up_list is None) else x_up_list
        y_up_list= self.y_up_list if (y_up_list is None) else y_up_list
        x_base_index= self.x_base_index if (x_base_index is None) else x_base_index
        n_vis_limit = len(c_list)-1 if n_vis_limit is None else n_vis_limit

        x_up_list_mu_ni= np.array([(xi.mean(axis=1).flatten().tolist() + [xi.shape[1]]) for xi in x_up_list])
        x_up_list_mu = ( x_up_list_mu_ni[:, :-1] * x_up_list_mu_ni[:, [-1]] ).sum(axis=0) /  x_up_list_mu_ni[:, [-1]].sum()


        # NOTE only use the first dimension of index set for visualization
        x_up_list_d0 = [ xi[...,0] for xi in x_up_list ]
        x_base_index_d0 = x_base_index[..., d0_index_set_plot]


        beta_threshold = self.get_max_precision_active_dynamic_beta(beta) if (beta_threshold is None) else beta_threshold

        d_index_set = self.x_base_index.shape[1]

        if x_up_d0_grid is None:
            x_up_grid_max = np.array([ z.max() for z in x_up_list_d0]).max()
            x_up_grid_min = np.array([ z.min() for z in x_up_list_d0]).min()
            x_up_d0_grid = np.linspace(start=x_up_grid_min, stop=x_up_grid_max,)
        # (n_grid, d_index_set), grid index set for plotting the results, default values are the center of the index set
        x_up_grid = np.ones((x_up_d0_grid.shape[0], d_index_set))
        for d in range(d_index_set):
            if d == d0_index_set_plot:
                x_up_grid[:, d] = x_up_d0_grid[:]
            else:
                index_center_d = self.x_base_index[:, d].mean()
                x_up_grid[:, d] *= index_center_d


        x_minmax = (np.min(x_up_d0_grid), np.max(x_up_d0_grid))
        x_base_index_repeat = np.array(x_base_index_d0.tolist() * len(self.kernel_list))
        n_basis_func = len( x_base_index_repeat )
        n_sample = len( y_up_list )


        fig,ax = plt.subplots(6, 1, figsize = (16, 60))


        k2k_active_beta= {}
        max_precision_active_dynamic_beta_ = self.get_max_precision_active_dynamic_beta(beta)
        for j,aj in enumerate(beta):
            if aj < max_precision_active_dynamic_beta_:
                k2k_active_beta[j] = len(k2k_active_beta)
        k_active_beta = sorted(list(k2k_active_beta.keys()))


        # precompute
        x_i_e_list_active, _ = self.get_update_z_e(
            z_bar= z_bar,
            beta= beta,
            sigma= sigma,
            active_only= True,
        )


        ln_like= self.get_ln_like( y_up_list = y_up_list, z_bar=z_bar, beta = beta, sigma = sigma, phi_xi_list_func=self.phi_xi_list_func )

        phi_active_grid = ( self.phi( x_up_grid, x_base_index= x_base_index ) ) [ k_active_beta, : ] if phi_active_grid is None else phi_active_grid
        mu_active = np.concatenate( x_i_e_list_active, axis=0)

        a_up_n05_active = np.diag( beta[ k_active_beta, None ].flatten()**(-0.5) )
        z_bar_active = z_bar[:, k_active_beta]


        # original and fitted functional data
        y_fit = (z_bar_active+mu_active) @ phi_active_grid
        y_mean = (z_bar_active) @ phi_active_grid
        for i in range(min(n_vis_limit, len(y_up_list)),):
            ax[0].scatter(
                x= x_up_list_d0[i].flatten(), y= y_up_list[i].flatten(), label=f'sample_{i}',
                s = ( plt.rcParams['lines.markersize'] ** 2 ) / 4,
                alpha= 0.9,
                )
            ax[1].scatter(
                x= x_up_list_d0[i].flatten(), y= y_up_list[i].flatten(), label=f'sample_{i}',
                s = ( plt.rcParams['lines.markersize'] ** 2 ) / 4,
                alpha= 0.9,
                )
            ax[1].plot( x_up_d0_grid, y_fit[i, :], label=f'sample_{i}_fit')
        ax[1].plot( x_up_d0_grid, y_mean[0], linestyle='--', label=f'mean_fit')


        # relevance vectors, estimation and truth
        rel_vec = [ (i,d) for i, d in enumerate( beta ) if d < beta_threshold] 
        for i, di in enumerate( rel_vec ):
            ax[0].axvline(x= x_base_index_repeat[di[0]], label= f'rel_vec_{i}@{x_base_index_repeat[di[0]]:.3e}' if i < n_vis_limit else '', c = 'g', ls= '-')
            ax[1].axvline(x= x_base_index_repeat[di[0]], label= f'rel_vec_{i}@{x_base_index_repeat[di[0]]:.3e}' if i < n_vis_limit else '', c = 'g', ls= '-')
            ax[3].axvline(x= x_base_index_repeat[di[0]], label= f'rel_vec_{i}@{x_base_index_repeat[di[0]]:.3e}' if i < n_vis_limit else '', c = 'g', ls= '-')


        # active kernel basis functions
        for i, (aj, j, jd) in enumerate( sorted( [ (beta[j], j, jd) for j,jd in k2k_active_beta.items() ] ) [: len(c_list)] ):
            ax[2].plot( x_up_d0_grid, phi_active_grid[jd, :], label=f'phi_active_{jd},beta={aj:.3e}@{x_base_index_repeat[j]:.3e}' if i < n_vis_limit else '', c = c_list[i % len(c_list)])
            ax[2].axvline(
                x= x_base_index_repeat[j], 
                ls= '-', color = c_list[i % len(c_list)],)


        # beta and weight
        ax4t = ax[3].twinx()
        m = np.zeros( shape= ( len( x_base_index_repeat ), len(y_up_list) ) )
        for j, ja in k2k_active_beta.items():
            m[j, : ] = mu_active.T[ ja, : ]
            ax[3].axvline(x= x_base_index_repeat[ j ], label= f'beta_index_{ja}', c = 'g', ls= '-')
        ax[3].scatter( 
            x = x_base_index_repeat, y = np.log10( beta ), label = 'lg(beta)', c = 'b',
            s = ( plt.rcParams['lines.markersize'] ** 2 ) / 4,
            alpha= 0.9,
            )
        for i in range(min(n_vis_limit, len(y_up_list)),):
            ax4t.scatter( 
                x = x_base_index_repeat, y = ( m[:, i] ), marker = 'x', label = f'weight_sample_{i}',
                s = ( plt.rcParams['lines.markersize'] ** 2 ) / 4,
                alpha= 0.9,
                )


        # spectrum analysis of the weight posterior
        ax5tx = ax[4].twinx()
        ax5txy = ax5tx.twiny()
        ax6tx = ax[5].twinx()
        ax6txy = ax6tx.twiny()

        phi_active_new = a_up_n05_active @ phi_active_grid
        u,s,v = np.linalg.svd( phi_active_new )
        stds = np.diag( phi_active_new @ phi_active_new.T )**0.5
        
        ax6txy.plot( s, label= 'std', marker='x' )
        components_stds = [
            (
                phi_active_new[ (i,), : ] / si * np.sign( phi_active_new[ (i,), np.argmax(np.abs(phi_active_new[ (i,), : ])) ]  ), 
                si
            )
            for i, si in enumerate( stds )
        ]
        components_stds = sorted( components_stds, key= lambda e:e[1], reverse= True)
        ax5txy.plot( [e[1] for e in components_stds], label= 'std', marker='x' )

        for i in range( min( len( components_stds ), n_vis_limit ) ):
            ui_phi, si_phi = components_stds[ i ]
            ax[4].plot( x_up_d0_grid, ui_phi.flatten(), label=f'component_{i}_(std={si_phi:.4e})')
            ax[5].plot( 
                x_up_d0_grid, 
                v[i].flatten() * np.sign( v[ (i,), np.argmax(np.abs(v[ (i,), : ])) ]  ), 
                label=f'component_{i}_(std={s[i]:.4e})')


        x_minmax_plot = (1.1 * x_minmax[0] - 0.1 * x_minmax[1],  1.1 * x_minmax[1] - 0.1 * x_minmax[0])
        ax[0].set_xlim(x_minmax_plot[0], x_minmax_plot[1])
        ax[1].set_xlim(x_minmax_plot[0], x_minmax_plot[1])
        ax[2].set_xlim(x_minmax_plot[0], x_minmax_plot[1])
        ax[3].set_xlim(x_minmax_plot[0], x_minmax_plot[1])
        ax4t.set_xlim(x_minmax_plot[0], x_minmax_plot[1])

        ax[0].set_title(f'n_basis_func={n_basis_func}, n_sample={n_sample}. center at [{",".join([f"{e:.3e}" for e in x_up_list_mu.tolist()])}]')
        ax[1].set_title(f'ln_like= {ln_like:.8e}, sigma_est={sigma:.8e}')

        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y_up_list')
        ax[0].legend()
        ax[0].grid()

        ax[1].set_xlabel('x')
        ax[1].set_ylabel('y')
        ax[1].legend()
        ax[1].grid()

        ax[2].set_xlabel('x')
        ax[2].set_ylabel(f'y')
        ax[2].legend()
        ax[2].grid()
        ax[2].set_title(f'active basis functions')

        ax[3].set_xlabel('x')
        ax[3].set_ylabel(f'lg(beta)')
        ax4t.set_ylabel(f'weight')
        ax[3].legend( loc='center left')
        ax4t.legend( loc='center right' )
        ax[3].grid()
        ax[3].set_title(f'beta and weight')

        ax[4].set_xlabel('x')
        ax[4].set_ylabel(f'y')
        ax[4].legend(loc='upper right')
        ax[4].grid()
        ax[4].set_title(f'normalized phi_active_new ( diag(β^(-0.5)) Φ )')

        ax6txy.set_xlabel('i_std')
        ax6tx.set_ylabel('std')
        ax6txy.legend( loc= 'upper center')

        ax5txy.set_xlabel('i_std')
        ax5tx.set_ylabel('std')
        ax5txy.legend( loc= 'upper center')

        ax[5].set_xlabel('x')
        ax[5].set_ylabel(f'y')
        ax[5].legend(loc='upper right')
        ax[5].grid()
        ax[5].set_title(f'SVD of phi_active_new ( diag(β^(-0.5)) Φ )')


        plt.tight_layout()
        
        if path_fig is not None:
            fig.savefig(path_fig)
        
        plt.close( fig )

        return path_fig
