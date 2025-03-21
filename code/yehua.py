'''
simulations in the paper selecting the number of components in funcional data analysis
'''
# %%
import os
import shutil
import numpy as np

from joblib import Parallel, delayed
import dill

from argparse import ArgumentParser
import pprint
import socket


import logging



def one_trial(
    random_seed,
    dir_out,

    eval_err_rt,
    n_log,
    m,
    snro,
    mu0,
    n_cluster_kernel,
    normalize_rbf,

    debug_mode,

    init_iter,
    n_iter_max,
    n_iter_max_delaywhite,
    init_dim,
    fix_sigma,
    use_fast,
    a0b0,
    tsim,
    coef_func_index,

    aba,
    tol_vb,
    t_ineff,
    max_cond_num_inverse,
    max_precision_active_alpha,
    max_precision_active_beta,

    use_pseudo,
    sigmak_init,
    sigmak_end,
    anneal_decay,

    loglevel,
    ):
    '''
    run yehua experiment, get the estimated number, copy the 2 summary figures
    '''

    np.random.seed( random_seed )

    logging.basicConfig(
        filename=f'{dir_out}/{random_seed}.log',
        filemode='a',
        level= loglevel,
        format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
        datefmt='%Y-%m-%d %A %H:%M:%S')


    try:
        # %%
        # number of functions
        n = 200

        # number of components
        p0 = 3

        # mean function
        mu = lambda t: 5 * (t-0.6)**2

        # eigen values
        w = np.array([0.6, 0.3, 0.1])

        sigma_u_sq = 0.2
        logging.critical(f'sigma_u_sq= {sigma_u_sq :.2e}')

        phi = lambda t : np.stack([
            np.ones_like(t), 
            2**0.5 * np.sin(2*np.pi*t), 
            2**0.5 * np.cos(2*np.pi*t),
            ], axis=0)

        ksi = np.random.randn( n, p0 ) @ np.diag( w**0.5 )


        if (snro == 1):
            pass

        elif (snro == 2):
            phi = lambda t : np.stack([
                np.ones_like(t), 
                2**0.5 * np.sin(2*np.pi*t), 
                2**0.5 * np.cos(4*np.pi*t),
                ], axis=0)

            ksi0 = np.random.randn( n, p0 ) @ np.diag( (w/3)**0.5 ) + (2*(w/3)**0.5)[None,:]
            ksi1 = np.random.randn( n, p0 ) @ np.diag( w**0.5 ) + (-(w/3)**0.5)[None,:]
            p_mix_ary = np.random.uniform(low=0.0, high=1.0, size=( n, p0 ))
            ksi = np.where( p_mix_ary <= 1/3, ksi0, ksi1 )

        elif (snro == 3):
            mu = lambda t: 12.5 * (t-0.5)**2 - 1.25

            phi = lambda t : np.stack([
                np.ones_like(t), 
                2**0.5 * np.cos(2*np.pi*t),
                2**0.5 * np.sin(4*np.pi*t),
                ], axis=0)

            w = np.array([4.0, 2.0, 1.0])

            sigma_u_sq = 0.5

            ksi = np.random.randn( n, p0 ) @ np.diag( w**0.5 )

        elif (snro == 4):

            mu = lambda t: 12.5 * (t-0.5)**2 - 1.25

            phi = lambda t : np.stack([
                np.ones_like(t), 
                2**0.5 * np.cos(2*np.pi*t),
                2**0.5 * np.sin(4*np.pi*t),
                ], axis=0)

            w = np.array([4.0, 2.0, 1.0])

            sigma_u_sq = 0.5

            ksi0 = np.random.randn( n, p0 ) @ np.diag( (w/3)**0.5 ) + (2*(w/3)**0.5)[None,:]
            ksi1 = np.random.randn( n, p0 ) @ np.diag( w**0.5 ) + (-(w/3)**0.5)[None,:]
            p_mix_ary = np.random.uniform(low=0.0, high=1.0, size=( n, p0 ))
            ksi = np.where( p_mix_ary <= 1/3, ksi0, ksi1 )

        elif (snro == 5):
            p0 = 6

            mu = lambda t: 12.5 * (t-0.5)**2 - 1.25

            w = np.array([4.0, 3.5, 3.0, 2.5, 2.0, 1.5])

            sigma_u_sq = 0.5

            ksi = np.random.randn( n, p0 ) @ np.diag( w**0.5 )

            phi = lambda t : np.stack([
                np.ones_like(t), 
                2**0.5 * np.sin(2*np.pi*t),
                2**0.5 * np.cos(2*np.pi*t),
                2**0.5 * np.sin(4*np.pi*t),
                2**0.5 * np.cos(4*np.pi*t),
                2**0.5 * np.sin(6*np.pi*t),
                ], axis=0)

        if mu0:
            # keep the shape of t
            mu = lambda t: t*0


        sigma_u = sigma_u_sq**(0.5)

        w_up_true= np.diag( w**0.5 )


        t_list = [
            np.sort(np.random.random(size=( m[int(np.around(np.random.rand(1)*(len(m)-1)))] )))[None,:] for i in range(n)
        ]
        phi_up_list = [
            phi(t_list[i].flatten()) for i in range(n)
        ]

        x_up_list = [
            ksi[i,...] @ phi_up_list[i] + mu( t_list[i] ) 
            for i,ti in enumerate(t_list)
        ]

        u_up_list = [
            sigma_u * np.random.randn(1, ti.shape[1]) for i,ti in enumerate(t_list)
        ]
        w_up_list = [
            x_up_list[i] + u_up_list[i] for i,ti in enumerate(t_list)
        ]


        # %%
        dir_debug = f'{dir_out}/{random_seed}/'
        if not os.path.exists(dir_debug):
            os.makedirs(dir_debug)
        
        # save data set for competing methods
        with open(f'{dir_out}/{random_seed}-xy.txt', 'w') as f:
            for i, ti in enumerate( t_list ):
                f.write( ' '.join( ti.flatten().astype(str).tolist() ) + '\n')
                f.write( ' '.join( w_up_list[i].flatten().astype(str).tolist() )  + '\n')
        
        # x1,x2: 1d numpy array
        cov = lambda x1,x2, phi= phi, w_up_true= w_up_true: phi( x1 ).T @ w_up_true**2 @ phi( x2 )

        with open(file=f'{dir_out}/{random_seed}-mu_cov_true.pkl', mode='wb') as f:
            dill.dump(
                obj= {
                    'cov': cov,
                    'mu': mu,
                },
                file= f,
            )


        if True:
            
            from util import get_kernel_list_and_coef_func_index
            if use_fast:
                from bsfda_fast import BSFDA
            else:
                from bsfda import BSFDA

            
            if not debug_mode:

                kernel_list, coef_func_index_ = get_kernel_list_and_coef_func_index(
                    x_up_list= t_list,
                    y_up_list= w_up_list,
                    n_cluster_kernel= n_cluster_kernel,
                    dir_debug= dir_debug,
                    normalize_rbf= normalize_rbf,
                    kernel_coef= 10,
                    )
                coef_func_index = coef_func_index_ if coef_func_index is None else coef_func_index

            else:

                t_min = np.min([np.min(t) for t in t_list])
                t_max = np.max([np.max(t) for t in t_list])
                t_domain_length = t_max - t_min

                from util import RBFKernel2, function_sample_scale_percentile, rbf_integral

                # %%
                # NOTE the basis functions should have the same scale as the data
                multipier_constant_value= function_sample_scale_percentile(function_sample_list= w_up_list, q=99) / 10
                logging.critical(f'multipier_constant_value= {multipier_constant_value}')


                ls_list = [0.0625, 0.08838835, 0.08838835, 0.08838835, 0.125, 0.1767767, 0.1767767, 0.1767767, 0.25, 0.35355339]

                coef_func_index= 14.491578628222499

                kernel_list= [
                    RBFKernel2(sigma=0, length_scale= c, multiplier= multipier_constant_value/rbf_integral(length_scale=c,a=t_domain_length/2) if normalize_rbf else multipier_constant_value )
                    for i,c in enumerate(ls_list)
                ]


            x_up_d0_grid = np.linspace(start=0,stop=1,num=1000)
            d0_index_set_plot = 0

            if a0b0 is None:
                vbfpca = BSFDA(
                    kernel_list= kernel_list,
                    precision_threshold_multiplier2min = t_ineff,
                    cond_num_max = max_cond_num_inverse,
                    max_precision_active_alpha = max_precision_active_alpha,
                    max_precision_active_beta = max_precision_active_beta,
                )
            else:
                vbfpca = BSFDA(
                    kernel_list= kernel_list,
                    a0= a0b0,
                    b0= a0b0,
                    precision_threshold_multiplier2min = t_ineff,
                    cond_num_max = max_cond_num_inverse,
                    max_precision_active_alpha = max_precision_active_alpha,
                    max_precision_active_beta = max_precision_active_beta,
                )
            vbfpca = vbfpca.fit(
                y_up_list= w_up_list,
                x_up_list= [t[...,None] for t in t_list],
                
                n_iter_max_beta_init= init_iter,

                param2eval_err_rt= {
                    'x_up_d0_grid': x_up_d0_grid,
                    'd0_index_set_plot': d0_index_set_plot,
                    'sigma_true': sigma_u,
                    'w_up_true': w_up_true,
                    'mean_true': lambda x: mu(x)[...,d0_index_set_plot],
                    'w_x_phi_up_list_true': lambda x: w_up_true @ phi( x )[...,d0_index_set_plot],
                } if eval_err_rt else None,
                n_log= n_log,
                n_active_basis_init= None if init_dim==-1 else init_dim,
                n_iter_max= n_iter_max,
                n_iter_max_delaywhite = n_iter_max_delaywhite,
                sim_threshold_new_basis= tsim,
                coef_func_index= coef_func_index,

                # NOTE sigma
                sigma_init= sigma_u if fix_sigma else None,
                sigma_is_fixed= fix_sigma,
                aba= aba,

                **({'use_pseudo': use_pseudo, 'sigmak_init': sigmak_init, 'sigmak_end': sigmak_end, 'anneal_decay': anneal_decay} if use_fast else {}),

                tol_convergence_variational= tol_vb * len(w_up_list),
                dir_debug = dir_debug,
            )
            path_dump=f'{dir_debug}/vbfpca.pkl'
            vbfpca.dump(path_dump=path_dump)
            with open(path_dump, 'rb') as f:
                vbfpca = dill.load(file=f)
            if (logging.root.level > logging.DEBUG):
                os.remove(path_dump)
            mu_mise, cov_mise = vbfpca.plot_results(
                x_up_d0_grid= x_up_d0_grid,
                d0_index_set_plot = d0_index_set_plot,

                sigma_true= sigma_u,
                w_up_true= w_up_true,
                mean_true = lambda x: mu(x)[...,d0_index_set_plot],
                w_x_phi_up_list_true= lambda x: w_up_true @ phi( x )[...,d0_index_set_plot],
                
                path_fig=f'{dir_debug}/bfpca_truth_vs_predict.pdf',
            )

            if not debug_mode:
                shutil.copy2(
                    src= f'{dir_debug}/length_scale_k_means.pdf',
                    dst= f'{dir_out}/{random_seed}-length_scale_k_means.pdf'
                )
            shutil.copy2(
                src= f'{dir_debug}/bfpca_truth_vs_predict.pdf',
                dst= f'{dir_out}/{random_seed}-bfpca_truth_vs_predict.pdf'
            )
            shutil.copy2(
                src= f'{dir_debug}/final.pdf',
                dst= f'{dir_out}/{random_seed}-final.pdf'
            )

            alpha_active = vbfpca.a_alpha_active/vbfpca.b_alpha_active
            threshold_alpha = min( (alpha_active.min() * vbfpca.precision_threshold_multiplier2min), vbfpca.max_precision_active_alpha )
            n_pc_estimate = np.sum( np.logical_not(np.logical_or(
                alpha_active > threshold_alpha,
                np.isclose( alpha_active, threshold_alpha, )
            )))


            if (logging.root.level >= logging.WARNING) and os.path.exists(dir_debug):
                shutil.rmtree(path= dir_debug)


            return n_pc_estimate, mu_mise, cov_mise

    except Exception as e:

        logging.exception(e)

        if (logging.root.level >= logging.ERROR) and os.path.exists(dir_debug):
            shutil.rmtree(path= dir_debug)

        return np.nan, np.nan, np.nan


def one_trial_f(param):
    return one_trial(**param)

if __name__ == "__main__":

    parser = ArgumentParser()

    # setting for data
    parser.add_argument("--snro", type=int, dest="snro", help="", required=True)
    parser.add_argument("--m", type=int, dest="m", nargs='+', help="number of points", default=5)
    parser.add_argument("--n_repeat", type=int, dest="n_repeat", help="number of repeats", default=200)
    parser.add_argument("--mu0", dest="mu0", help="whether the true mean is 0", action='store_true')
    # setting for optimization
    parser.add_argument("--n_ck", type=int, dest="n_ck", help="", default=1)
    parser.add_argument("--init_iter", type=int, dest="init_iter", help="number of points", default=100)
    parser.add_argument("--init_dim", type=int, dest="init_dim", help="number of basis function in variational inference after relevance vector machine is done", default=None)
    parser.add_argument("--n_iter_max", type=int, dest="n_iter_max", help="maximum number of iterations, default is that of vbfpca", default=None)
    parser.add_argument("--n_iter_max_delaywhite", type=int, dest="n_iter_max_delaywhite", help="maximum number of iterations for delay white, default is that of vbfpca", default=1000)
    parser.add_argument("--tsim", type=float, dest="tsim", help="threshold for cosine similarity", default=0.99)
    parser.add_argument("--tol_vb", type=float, dest="tol_vb", help="tolerance multiplier for convergence of variational inference", default=1e-7)
    parser.add_argument("--t_ineff", type=float, dest="t_ineff", help="threshold multiplier for ineffective dimension", default=1e5)
    parser.add_argument("--max_precision_active_alpha", type=float, dest="max_precision_active_alpha", help="maximum precision for scale parameters for active dimensions, default is None", default=1e5)    
    parser.add_argument("--max_precision_active_beta", type=float, dest="max_precision_active_beta", help="maximum precision for scale parameters for active dimensions, default is None", default=1e5)    
    parser.add_argument("--coef_func_index", type=int, dest="coef_func_index", help="coef_func_index", default=None)
    # setting for debugging
    parser.add_argument("--dir_out", type=str, dest="dir_out", help="", required=True)
    parser.add_argument("--n_job", type=int, dest="n_job", help="number of cores", default=50)
    parser.add_argument("--debug_mode", help="mode of debugging", action='store_true')
    parser.add_argument("--loglvl", type=int, dest="loglvl", help="", default=logging.WARNING)
    parser.add_argument("--n_log", type=int, dest="n_log", help="", default=int(1e2))
    parser.add_argument("--eval_err_rt", help="calculate errors during optimization in real time", action='store_true')
    # experimental setting that is not used in the paper
    parser.add_argument("--max_cond_num_inverse", type=float, dest="max_cond_num_inverse", help="maximum condition number for inverting matrices, used for scaled Tikhonov-regularization, default is None", default=None)
    parser.add_argument("--fix_sigma", help="fix sigma at the true value", action="store_true")
    parser.add_argument("--use_fast", help="use the faster algorithm with column independence", action="store_true")
    parser.add_argument("--aba", type=int, dest="aba", help="a for alpha and beta", default=None)
    parser.add_argument("--a0b0", type=float, dest="a0b0", help="a0 and b0", default=None)



    # peudo_new_test1_sigmakn2_n3_anl10kfix1_exp_it10k_tolran7_rterr
    # whether use pseudo data
    parser.add_argument("--use_pseudo", help="use pseudo data", action='store_true')
    parser.add_argument("--sigmak_init", type=float, dest="sigmak_init", help="initial sigma_k", default=1e-2)
    parser.add_argument("--sigmak_end", type=float, dest="sigmak_end", help="end sigma_k", default=1e-3)
    parser.add_argument("--anneal_decay", type=str, dest="anneal_decay", help="decay of sigma_k, exp or linear or log", default='exp')


    '''
    example:
    nohup python -u code/bsfda/yehua.py --snro 1 --m 5 --n_repeat 20 --n_ck 10 --n_job 20 --dir_out code/bsfda/data/yehua/test/s1/m5/nck10_test >> code/bsfda/data/yehua/test/s1/m5/nck10_test.log &
    '''

    args = parser.parse_args()

    logging.basicConfig(
        level= args.loglvl,
        format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
        datefmt='%Y-%m-%d %A %H:%M:%S')

    logging.critical('args is \n%s'%(pprint.pformat(args.__dict__, indent=4)))

    logging.critical(f'hostname= {socket.gethostname()}')

    dir_out = os.path.abspath(args.dir_out)
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    np.random.seed(31415926)

    n_repeat = args.n_repeat
    n_job = args.n_job

    rand_seed_list = list(range(n_repeat))

    p_list = [
        {
            'random_seed': s,
            'dir_out': dir_out,
            'snro': args.snro,
            'm': args.m,
            'mu0': args.mu0,
            
            'eval_err_rt': args.eval_err_rt,
            'n_log': args.n_log,
            'n_cluster_kernel': args.n_ck,
            'normalize_rbf': True,

            'init_iter': args.init_iter,
            'n_iter_max': args.n_iter_max,
            'n_iter_max_delaywhite': args.n_iter_max_delaywhite,
            'init_dim': args.init_dim,
            'fix_sigma': args.fix_sigma,
            'use_fast': args.use_fast,
            'a0b0': args.a0b0,
            'tsim': args.tsim,
            'aba': args.aba,
            'tol_vb': args.tol_vb,
            't_ineff': args.t_ineff,
            'max_cond_num_inverse': args.max_cond_num_inverse,
            'max_precision_active_alpha': args.max_precision_active_alpha,
            'max_precision_active_beta': args.max_precision_active_beta,
            'coef_func_index': args.coef_func_index,

            'use_pseudo': args.use_pseudo,
            'sigmak_init': args.sigmak_init,
            'sigmak_end': args.sigmak_end,
            'anneal_decay': args.anneal_decay,

            'loglevel': args.loglvl,
            'debug_mode': args.debug_mode,
        } for s in rand_seed_list
    ]
    if args.debug_mode:
        estimate_npc_mumise_covmise = [one_trial_f(p_list[0])]
    else:
        estimate_npc_mumise_covmise = Parallel(n_jobs=min(n_job, len(p_list)))(delayed(one_trial_f)(p) for p in p_list)

    estimate_npc_mumise_covmise = np.array(estimate_npc_mumise_covmise)

    logging.critical(f'estimate_npc_mumise_covmise = {estimate_npc_mumise_covmise}, average= {np.nanmean(estimate_npc_mumise_covmise, axis=0)}')
    logging.critical("NOTE, the following confidence interval has ± 2 standard deviation, and the standard deviation uses Bessel's correction.")
    logging.critical(f'n_total={ estimate_npc_mumise_covmise.shape[0] }, n_valid= {np.sum( np.logical_not( np.isnan(estimate_npc_mumise_covmise[:,0]) ) )}')
    logging.critical(f'rank= {np.nanmean(estimate_npc_mumise_covmise[:,0])}±{2*np.nanstd(estimate_npc_mumise_covmise[:,0], ddof=1)}')
    logging.critical(f'mu_amise= {np.nanmean(estimate_npc_mumise_covmise[:,1])}±{2*np.nanstd(estimate_npc_mumise_covmise[:,1], ddof=1)}')
    logging.critical(f'cov_amise= {np.nanmean(estimate_npc_mumise_covmise[:,2])}±{2*np.nanstd(estimate_npc_mumise_covmise[:,2], ddof=1)}')
    logging.critical(f'median cov_mise i= {np.argmin( np.abs( estimate_npc_mumise_covmise[:,2] - np.nanmedian(estimate_npc_mumise_covmise[:,2]) ) )}')

    np.savetxt(
        fname=f'{dir_out}/estimate_npc_mumise_covmise.txt',
        X= estimate_npc_mumise_covmise,
        fmt='%s',
    )



