# %%
import os
# NOTE TODO to avoid "Segmentation fault      (core dumped)" with the warning: "OpenBLAS warning: precompiled NUM_THREADS exceeded, adding auxiliary array for thread metadata." on our server, we need to set the following environment variables
os.environ["OPENBLAS_NUM_THREADS"] = "64"  # Adjust based on your needs
os.environ["OMP_NUM_THREADS"] = "64"

import socket
import numpy as np
import logging
from argparse import ArgumentParser
import os
import pprint

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

cores = os.cpu_count()
logging.critical(f'number of cores= {cores}')

from util import RBFKernel2, cv_kernel_kmeans

# %%
def main(
    dir_out,
    tol_conv_coef,
    # number of functions
    n = 20,
    # number of measurements
    m = [10],
    # dimension of index set
    d_index_set = 2,
    n_cluster_kernel = 1,
    init_iter = 100,
    n_iter_max = 10000,
    coef_func_index = None,
    n_log = 1,
    eval_err_rt = False,
    mu_type = '0',
    tsim = 0.99,
    use_true_gaussian_length_scale = False,
    ls = 1/12,

    use_fast = False,
    n_iter_max_delaywhite = 1000,
    ):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)


    if use_fast:
        from bsfda_fast import BSFDA
    else:
        from bsfda import BSFDA

    # mean function, (n, d_index_set) -> (n, )
    if mu_type is None:
        mu = lambda t: 5 * ((t-0.6)**2).sum(axis=1)
    elif mu_type == '0':
        mu = lambda t: np.zeros_like(t[...,0])



    # eigen values
    w = np.array([0.6, 0.3, 0.4])

    logging.critical(f'w={w}')

    sigma_u_sq = 0.2


    def phi_gaussian(t):
        '''
        t: (n, d_index_set)
        g(t) = exp( -0.5 * || t / ls ||_2^2 )
        \int g(t)^2 dt = \int exp( - || t / ls ||_2^2 ) dt
            = \int exp( - 0.5 * || t / ( 0.5^0.5 * ls ) ||_2^2 ) dt
            = \int exp( - 0.5 * || t / ls_new ||_2^2 ) dt
        ls_new = 0.5^0.5 * ls
        (2 * \pi * ls_new^2 ) ^ (-d_index_set/2) * \int g(t)^2 dt = 1
            = ( \pi * ls^2 ) ^ (-d_index_set/2) * \int g(t)^2 dt
        ---
        return: (p0, n)
        '''
        center_list = [
            np.array( [0.5] * d_index_set )[None,:]
            for i in range(3)
        ]
        center_list[1][0,0] = 0.4
        center_list[1][0,1] = 0.4
        center_list[1][0,2] = 0.4
        center_list[1][0,3] = 0.4
        center_list[2][0,0] = 0.6
        center_list[2][0,1] = 0.6
        center_list[2][0,2] = 0.6
        center_list[2][0,3] = 0.6

        phi1 = ( np.pi * ls**2 )**(-d_index_set/2) * np.exp(-0.5*np.linalg.norm( (t - center_list[0])/ls , axis=1)**2)
        phi2 = ( np.pi * ls**2 )**(-d_index_set/2) * np.exp(-0.5*np.linalg.norm( (t - center_list[1])/ls , axis=1)**2)
        phi3 = ( np.pi * ls**2 )**(-d_index_set/2) * np.exp(-0.5*np.linalg.norm( (t - center_list[2])/ls , axis=1)**2)
        return np.stack([phi1-phi2, phi2, phi3], axis=0)



    phi = phi_gaussian

    # number of components
    p0 = phi(np.zeros((1,d_index_set))).shape[0]
    logging.info(f'p0={p0}')

    ksi = np.random.randn( n, p0 ) @ np.diag( w**0.5 )


    sigma_u = sigma_u_sq**(0.5)

    w_up_true= np.diag( w**0.5 )


    x_up_list = [
        np.random.random(size=( m[int(np.around(np.random.rand(1)*(len(m)-1)))], d_index_set ))[None,:] for i in range(n)
    ]
    phi_up_list = [
        phi(x_up_list[i][0]) for i in range(n)
    ]

    f_up_list = [
        ksi[[i],...] @ phi_up_list[i] + mu( x_up_list[i][0] ) 
        for i,ti in enumerate(x_up_list)
    ]

    e_up_list = [
        sigma_u * np.random.randn(1, ti.shape[1]) for i,ti in enumerate(x_up_list)
    ]
    y_up_list = [
        f_up_list[i] + e_up_list[i] for i,ti in enumerate(x_up_list)
    ]


    
    a0b0 = None
    multipier_constant_value = 1


    if use_true_gaussian_length_scale:
        ls_list = [ls]
    else:
        ls_list = cv_kernel_kmeans(
            length_scale_list= [
                2**( 4 - c/2)
                for c in range(17)
            ],
            n_clusters=n_cluster_kernel,
            x_up_list= x_up_list,
            y_up_list= y_up_list, 
            path_fig=f'{dir_out}/length_scale_k_means.pdf',
            multipier_constant_value= multipier_constant_value,
            normalize_rbf= False,
        )
    logging.critical(f'ls_list={ls_list}')


    kernel_list= [
        RBFKernel2(sigma=0, length_scale= c, multiplier= multipier_constant_value )
        for i,c in enumerate(ls_list)
    ]


    if a0b0 is None:
        vbfpca = BSFDA(
            kernel_list= kernel_list,
            max_precision_active_alpha = np.float64(1e6),
            max_precision_active_beta = np.float64(1e6),
        )
    else:
        vbfpca = BSFDA(
            kernel_list= kernel_list,
            a0= np.float64(a0b0),
            b0= np.float64(a0b0),
            max_precision_active_alpha = np.float64(1e6),
            max_precision_active_beta = np.float64(1e6),
        )
    vbfpca = vbfpca.fit(
        y_up_list= y_up_list,
        x_up_list= [t for t in x_up_list],
        
        n_iter_max_beta_init= init_iter,

        param2eval_err_rt= {
            'sigma_true': sigma_u,
            'w_up_true': w_up_true,
            'mean_true': lambda x, mu=mu: mu(x),
            'w_x_phi_up_list_true': lambda x, w_up_true=w_up_true, phi=phi: w_up_true @ phi( x ),
            'full_cov_mise_npts': 10,
        } if eval_err_rt else None,
        n_log= n_log,
        n_iter_max= n_iter_max,
        n_iter_max_delaywhite= n_iter_max_delaywhite,
        sim_threshold_new_basis= tsim,
        coef_func_index= coef_func_index,
        tol_converged_beta_init = tol_conv_coef * len(x_up_list) * 10,
        tol_convergence_variational= tol_conv_coef * len(x_up_list),

        dir_debug = dir_out,

        # NOTE this should be used for the case of the fast algorithm
        sigmak_init= 1e-2,
        sigmak_end= 1e-5,
    )

    for d0 in range(d_index_set):
        mu_mise_d0, cov_mise_d0 = vbfpca.plot_results(
            d0_index_set_plot= d0,

            sigma_true= sigma_u,
            w_up_true= w_up_true,
            w_x_phi_up_list_true = lambda x: w_up_true @ phi( x ),
            mean_true = lambda x: mu(x),

            path_fig = f'{dir_out}/results_d{d0}.pdf',
        )
        logging.critical(f'mu_mise_d{d0}={mu_mise_d0}, cov_mise_d{d0}={cov_mise_d0}')

# %%
if __name__ == '__main__':
    parser = ArgumentParser()
    # setting for simulation
    parser.add_argument('--dir_out', type=str)
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument("--m", type=int, dest="m", nargs='+', help="number of points", default=5)
    parser.add_argument('--d_index_set', type=int, default=2)
    parser.add_argument('--mu_type', type=str, default='0')
    parser.add_argument('--ls', type=float, default=1/12)
    # hyper parameter
    parser.add_argument('--n_cluster_kernel', type=int, default=1)
    parser.add_argument('--init_iter', type=int, default=100)
    parser.add_argument('--n_iter_max', type=int, default=10000)
    parser.add_argument('--coef_func_index', type=int, default=None)
    parser.add_argument('--tsim', type=float, default=0.99)
    parser.add_argument('--use_true_gaussian_length_scale', default=False, action='store_true')
    parser.add_argument('--tol_conv_coef', type=float, default=2e-5)
    parser.add_argument("--use_fast", help="use the faster algorithm with column independence", action="store_true")
    parser.add_argument("--n_iter_max_delaywhite", type=int, dest="n_iter_max_delaywhite", help="maximum number of iterations for delay white, default is that of vbfpca", default=1000)
    # setting for logging
    parser.add_argument('--n_log', type=int, default=1)
    parser.add_argument("--eval_err_rt", help="calculate errors during optimization in real time", action='store_true')


    args = parser.parse_args()

    logging.critical(f'args={pprint.pformat(args.__dict__)}')

    np.random.seed(31415926)

    logging.critical(f'hostname= {socket.gethostname()}, OPENBLAS_NUM_THREADS= {os.environ["OPENBLAS_NUM_THREADS"]}, OMP_NUM_THREADS= {os.environ["OMP_NUM_THREADS"]}')

    main(
        dir_out= args.dir_out,
        n= args.n,
        m= args.m,
        d_index_set= args.d_index_set,
        n_cluster_kernel= args.n_cluster_kernel,
        init_iter= args.init_iter,
        n_iter_max= args.n_iter_max,
        coef_func_index= args.coef_func_index,
        n_log= args.n_log,
        eval_err_rt= args.eval_err_rt,
        mu_type= args.mu_type,
        tsim= args.tsim,
        use_true_gaussian_length_scale= args.use_true_gaussian_length_scale,
        ls= args.ls,
        tol_conv_coef= args.tol_conv_coef,
        use_fast= args.use_fast,
        n_iter_max_delaywhite= args.n_iter_max_delaywhite,
    )