# %%
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
    datefmt='%Y-%m-%d %A %H:%M:%S')

import os
# get number of cores using os
cores = os.cpu_count()
# n_thread = max( 1, min( int( cores * 0.95 ), cores - 1 ) )
n_thread = 24
logging.info(f'Number of cores: {cores}, number of threads: {n_thread}')
os.environ["NUMEXPR_MAX_THREADS"] = str( n_thread )

import json
import glob
import dill
import numexpr as ne
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pprint
import time

# numpy mkl thread, mkl_set_num_threads_local
import mkl
mkl.set_num_threads( n_thread )

from sklearn.cluster import AgglomerativeClustering

from util import get_kernel_list_and_coef_func_index, KernelPresLatLonTime
from bsfda_fast import BSFDA, load_bsfda
from argo_vis import plot_interpolated_temperature
from util import lotlong2dist

np.random.seed(31415926)

# %%
numexpr_num_threads = int( os.environ.get('NUMEXPR_MAX_THREADS', 1) )
ne.set_num_threads(numexpr_num_threads)
ne.set_vml_num_threads(numexpr_num_threads)
ne.set_vml_accuracy_mode('low')
pd.set_option('compute.use_numexpr', True)
logging.info(f'{np.show_config()}')



# %%
# arguements parsing
parser = argparse.ArgumentParser(description='BSFDA on Argo data')
parser.add_argument('--ls_geo', type=float, default= 2000, help='length scale for geographical distance')
parser.add_argument('--ls_time', type=float, default= 3, help='length scale for time')
parser.add_argument('--ls_pres', type=float, default= 70, help='length scale for pressure')
parser.add_argument('--rvmbs', type=int, default= 1000, help='batch size for NRVM')
parser.add_argument('--itmax', type=int, default= 20000, help='maximum number of iterations')
parser.add_argument('--itrvm', type=int, default= 300, help='maximum number of iterations for RVM')
parser.add_argument('--nb', type=int, default= 2000, help='total number of basis functions')
parser.add_argument('--test', action='store_true', help='whether to use test mode')
# start year
parser.add_argument('--syear', type=int, default= 1998, help='start year')
# condition number of pseudo inverse
parser.add_argument('--rcond_pinv', type=float, default= 1e-10, help='condition number of pseudo inverse')

args, unknown = parser.parse_known_args()
logging.info(f'Arguments: {pprint.pformat(vars(args))}')

nrvm_batch_size = args.rvmbs


# %%
dir_root = './data/argo'
dir_cache = f'{dir_root}/float/.cache'
depth_range = (0, 200)
force_float64 = True

test_ratio = 0.1

if not os.path.exists(dir_cache):
    os.makedirs(dir_cache)
with open(f'{dir_cache}/argo_data.pkl', 'rb') as f:
    x_up_list, y_up_list, year_list = dill.load(f)

x_up_list_, y_up_list_, year_list_ = [], [], []
for i, y in enumerate(year_list):
    if y >= args.syear:
        x_up_list_.append(x_up_list[i])
        y_up_list_.append(y_up_list[i])
        year_list_.append(y)
x_up_list, y_up_list, year_list = x_up_list_, y_up_list_, year_list_

# shuffle data
for i in range(len(x_up_list)):
    idx = np.random.permutation(x_up_list[i].shape[1])
    x_up_list[i] = x_up_list[i][:,idx]
    y_up_list[i] = y_up_list[i][:,idx]

# %%
is_test = args.test
dir_debug= f'{dir_root}/out'
logging.info(f'dir_debug: {dir_debug}')

if not os.path.exists(dir_debug):
    os.makedirs(dir_debug)

length_scale_geo = 4e4 / 4
length_scale_time = 2
length_scale_pres = 200 / 4
period_time = 1
kernel_multiplier = 1

# %%
# plot the measured temperature
year_ = 2021
year_to_inspect_ = np.argwhere( np.array(year_list) == year_ )[0][0]
plot_interpolated_temperature(
    train_points= np.hstack( ( x_up_list[year_to_inspect_][0,:,:4], y_up_list[year_to_inspect_][0][:,None] ) ),
    # february
    ref_point= np.array([0.0, 0.0, 0.0, 45/365, np.inf]),
    
    y = year_list[year_to_inspect_],
    path_to_save=f'{dir_debug}/data_temperature_map_{year_}.pdf',
    depth_range=depth_range,

    marker_size_range = (10, 10),

    threshold_depth = 5,
    threshold_time = 15/365,
    threshold_geodetic = 50,
    no_edges=True,
    # dummy arguments
    lat_lon_intp=np.ones((2,2), dtype=x_up_list[0].dtype)*np.nan,
    dept_time_intp=np.ones((2,2), dtype=x_up_list[0].dtype)*np.nan,
    n_grid=2,
)

# %%
x_up_list, y_up_list = [], []
x_up_list_test, y_up_list_test = [], []
for i, x in enumerate(x_up_list_):
    n = (x.shape[1] * 1e-3) if is_test else (x.shape[1] * 1e0)
    n = int(max(args.nb*3, n))

    p_id = x[0,:,-2].astype(int)
    p_id_unique = np.unique(p_id)
    np.random.shuffle(p_id_unique)

    n_test_profile = int(len(p_id_unique) * test_ratio)
    p_id_test = p_id_unique[:n_test_profile]
    p_id_train = p_id_unique[n_test_profile:]

    ix_test = np.where( np.isin(p_id, p_id_test) )[0]
    ix_train = np.where( np.isin(p_id, p_id_train) )[0][:n]

    logging.info(f'year {year_list_[i]}: n_test_profile={n_test_profile}, n_test={len(ix_test)}, n_train_profile={len(p_id_train)}, n_train={len(ix_train)}')

    # n_test = int(n * test_ratio)
    # n_tain = n - n_test

    x_up_list.append(x[:, ix_train,:4])
    y_up_list.append(y_up_list_[i][:, ix_train])
    x_up_list_test.append(x[:, ix_test,:4])
    y_up_list_test.append(y_up_list_[i][:, ix_test])
    
    
# total number of training points
logging.info(f'total number of training points: {sum([x.shape[1] for x in x_up_list])}')

# center y_up_list
# %%
y_up_all_mean = 0
y_up_all_std = 1

    
def interpolate_temperature(
    i,
    x,
    vbfda,
    ):
    '''
    i: int
        index of the function to interpolate
    x: np.ndarray, location to interpolate
        shape (n_points, n_dim)
    '''
    vbfpca = vbfda

    k_active_list = vbfpca.k_active_list
    beta_active_expect = vbfpca.a_beta_active / vbfpca.b_beta_active
    threshold_beta = min( (beta_active_expect.min() * vbfpca.precision_threshold_multiplier2min), vbfpca.max_precision_active_beta )
    idx_beta_active_effective = np.where( beta_active_expect<threshold_beta )[0].tolist()
    if len(idx_beta_active_effective)<1:
        idx_beta_active_effective = [np.argmin(beta_active_expect)]
        logging.warning(f'all betas are too large and the smallest one will be used for plotting!')


    year_to_inspect = i
    lat_lon_grid = x


    mu_zbar_effective = vbfpca.mu_zbar_active[ :, idx_beta_active_effective ]


    k_effective_list = np.array(k_active_list)[ idx_beta_active_effective ].tolist()
    n_index_basis = vbfpca.x_base_index.shape[0]

    phi_up_grid_latlon_effective = []
    for k in k_effective_list:
        x_up_basis_k = vbfpca.x_base_index[ [k % n_index_basis] ]
        phi_up_grid_latlon_effective.append( vbfpca.kernel_list[ k // n_index_basis ] ( x_up_basis_k, lat_lon_grid ) )
    phi_up_grid_latlon_effective = np.concatenate( phi_up_grid_latlon_effective, axis=0 )

    theta_i = vbfpca.mu_theta_i_list_active[year_to_inspect][:,idx_beta_active_effective]


    intp = ( mu_zbar_effective + theta_i ) @ phi_up_grid_latlon_effective

    # recover scale
    intp = intp * y_up_all_std + y_up_all_mean

    return intp


# %%
def objf(
    length_scale_geo = args.ls_geo,
    length_scale_time = args.ls_time,
    length_scale_pres = args.ls_pres,
    rcond_pinv = args.rcond_pinv,
    dir_debug=None,
    ):
    kernel_multiplier = 1e1
    period_time = 1

    k4d = KernelPresLatLonTime(
        sigma=0, 
        length_scale_geo= np.float64( length_scale_geo ) if force_float64 else length_scale_geo, 
        length_scale_time= np.float64( length_scale_time ) if force_float64 else length_scale_time,
        multiplier= np.float64( kernel_multiplier ) if force_float64 else kernel_multiplier, 
        length_scale_pres= np.float64( length_scale_pres ) if force_float64 else length_scale_pres,
        period_time= np.float64( period_time ) if force_float64 else period_time
        )
    coef_func_index = args.nb / np.max([
                        xi.shape[1] for xi in x_up_list
                    ])

    
    
    '''# %%'''
    vbfpca = BSFDA(
        kernel_list= [k4d], 
        EPS=np.finfo(y_up_list[0].dtype).eps,
        cond_num_max= None, 
        rcond_pinv= rcond_pinv,
        ).fit(
            y_up_list= y_up_list,
            x_up_list= x_up_list,

            coef_func_index= coef_func_index.astype(np.float64) if force_float64 else coef_func_index,

            n_log=1000,
            dir_debug = dir_debug,

            sigma_init= np.float64(1) if force_float64 else 1,
            sigma_is_fixed = False,
            sigma_is_fixed_rvm = True,

            nrvm_batch_size= nrvm_batch_size,

            nrvm_llkh = False,
            refit_nrvm= False,
            nrvm_cond_num_max= None,

            n_iter_max_delaywhite= 500 if not is_test else 50,

            # test
            n_iter_max_beta_init= args.itrvm,
            n_iter_max= args.itmax,
            n_basis_add_serial= 1,
    )
    if dir_debug is not None:
        vbfpca.plot_results(
            path_fig=f'{dir_debug}/bfpca_truth_vs_predict.pdf',
        )
        for d0 in range( x_up_list[0].shape[2] ):
            vbfpca.plot_results(
                d0_index_set_plot= d0,
                path_fig=f'{dir_debug}/results_d{d0}.pdf',
        )
        vbfpca.dump(path_dump=f'{dir_debug}/vbfpca.pkl', exclude_data=True)



    # Calculate prediction errors across all years 
    # errors = []
    # y_true = []
    errors, y_true = [], []
    errors_test, y_true_test = [], []

    for year_idx in range(len(x_up_list)):
        logging.info(f'Predicting for year {year_list[year_idx]}')
        # Get predictions for this year
        y_pred = interpolate_temperature(year_idx, x_up_list[year_idx][0], vbfpca)
        y_pred_test = interpolate_temperature(year_idx, x_up_list_test[year_idx][0], vbfpca)
        
        # Get true values
        y_actual = y_up_list[year_idx][0] 
        y_actual_test = y_up_list_test[year_idx][0]
        
        # Store errors and true values
        e = (y_pred.flatten() - y_actual.flatten()).tolist()
        # errors += e
        errors.extend(e)
        y_true.extend(y_actual)
        e_test = (y_pred_test.flatten() - y_actual_test.flatten()).tolist()
        errors_test.extend(e_test)
        y_true_test.extend(y_actual_test)

        logging.info(f'RMSE Train = {np.sqrt(np.mean(np.square(e))):.4f}, Test = {np.sqrt(np.mean(np.square(e_test))):.4f}')

    # Convert to arrays
    errors = np.array(errors).flatten()
    y_true = np.array(y_true)
    errors_test = np.array(errors_test).flatten()
    y_true_test = np.array(y_true_test)

    # Calculate metrics
    rmse = np.sqrt(np.mean(np.square(errors)))
    y_mean = np.mean(y_true)
    r2 = 1 - np.sum(np.square(errors)) / np.sum(np.square(y_true - y_mean))

    rmse_test = np.sqrt(np.mean(np.square(errors_test)))
    y_mean_test = np.mean(y_true_test)
    r2_test = 1 - np.sum(np.square(errors_test)) / np.sum(np.square(y_true_test - y_mean_test))

    # Log results
    logging.info(f'Oveall RMSE Train: {rmse:.4f}, Test: {rmse_test:.4f}. R^2 Train: {r2:.4f}, Test: {r2_test:.4f}')

    return rmse_test


# %%
objf(
    length_scale_geo = args.ls_geo,
    length_scale_time = args.ls_time,
    length_scale_pres = args.ls_pres,
    # length_scale_time_rbf = args.ls_time_rbf,
    dir_debug=dir_debug,
)


# %%
vbfpca = load_bsfda(f'{dir_debug}/vbfpca.pkl')
for year in [2021]:

    k_active_list = vbfpca.k_active_list
    n_basis_active = len(k_active_list)

    w_up_active = vbfpca.mu_vecwup_active.reshape((n_basis_active, n_basis_active)).T

    # idx_alpha_active_effective
    alpha_active_expect = vbfpca.a_alpha_active / vbfpca.b_alpha_active
    threshold_alpha = min( (alpha_active_expect.min() * vbfpca.precision_threshold_multiplier2min), vbfpca.max_precision_active_alpha )
    idx_alpha_active_effective = np.where(
        np.logical_not(np.logical_or(
            alpha_active_expect > threshold_alpha,
            np.isclose(alpha_active_expect, threshold_alpha),
        ))
    )[0].tolist()
    if len(idx_alpha_active_effective)<1:
        idx_alpha_active_effective = [np.argmin(alpha_active_expect)]
        logging.warning(f'all alphas are too large and the smallest one will be used for plotting!')

    # idx_beta_active_effective
    beta_active_expect = vbfpca.a_beta_active / vbfpca.b_beta_active
    threshold_beta = min( (beta_active_expect.min() * vbfpca.precision_threshold_multiplier2min), vbfpca.max_precision_active_beta )
    idx_beta_active_effective = np.where( beta_active_expect<threshold_beta )[0].tolist()
    if len(idx_beta_active_effective)<1:
        idx_beta_active_effective = [np.argmin(beta_active_expect)]
        logging.warning(f'all betas are too large and the smallest one will be used for plotting!')


    # 2016 february 
    # Select a specific year to inspect
    year_to_inspect = np.argwhere( np.array(year_list) == year )[0][0]
    # i_inspect = 5807

    selected_data_points = [0, 0, -30, 5/12]

    # Check if data points are available
    # Extract z and y values
    # match selected_data_points to the closest point in the dataset
    tp = np.hstack( ( x_up_list[year_to_inspect][0], y_up_list[year_to_inspect][0][:,None] * y_up_all_std + y_up_all_mean ) )
    # normailze
    # tp_s, tp_m = tp.std(axis=0), tp.mean(axis=0)
    tp_s = np.array( [200, 20000**0.5, 20000**0.5, 1 ] )
    selected_data_points_norm = ( np.array(selected_data_points ) ) / tp_s
    tp_norm = ( tp[:,:4] ) / tp_s
    i_ = np.argmin( np.linalg.norm( tp_norm - selected_data_points_norm, axis=1 ) )
    selected_data_points = tp[i_].tolist()
    selected_data_points = [ selected_data_points[:4], selected_data_points[4] ]


    dept, lat, lon, t = selected_data_points[0]
    temp = selected_data_points[1]

    z = vbfpca.mu_zi_list_active[year_to_inspect]
    zbar = vbfpca.mu_zbar_active 

    # grid for whole globe
    n_grid = 300
    lat_lon_grid = np.array([
        [dept, lt, ln, t]
        for lt in np.linspace(-90, 90, n_grid)
        for ln in np.linspace(-180, 180, n_grid)
    ])
    # grid for depth and time
    dept_time_grid = np.array([
        [dp, lat, lon, tm]
        for dp in np.linspace(depth_range[0], depth_range[1], n_grid)
        for tm in np.linspace(0, 1, n_grid)
    ])


    mu_zbar_effective = vbfpca.mu_zbar_active[ :, idx_beta_active_effective ]
    z_effective = vbfpca.mu_zi_list_active[year_to_inspect][:,idx_alpha_active_effective]


    k_effective_list = np.array(k_active_list)[ idx_beta_active_effective ].tolist()
    n_index_basis = vbfpca.x_base_index.shape[0]


    # time this step
    st = time.time()
    phi_up_grid_latlon_effective = []
    phi_up_grid_depttime_effective = []
    for k in k_effective_list:
        x_up_basis_k = vbfpca.x_base_index[ [k % n_index_basis] ]
        phi_up_grid_latlon_effective.append( vbfpca.kernel_list[ k // n_index_basis ] ( x_up_basis_k, lat_lon_grid ) )
        phi_up_grid_depttime_effective.append( vbfpca.kernel_list[ k // n_index_basis ] ( x_up_basis_k, dept_time_grid ) )
    phi_up_grid_latlon_effective = np.concatenate( phi_up_grid_latlon_effective, axis=0 )
    phi_up_grid_depttime_effective = np.concatenate( phi_up_grid_depttime_effective, axis=0 )

    lat_lon_intp = ( mu_zbar_effective + vbfpca.mu_theta_i_list_active[year_to_inspect][:,idx_beta_active_effective] ) @ phi_up_grid_latlon_effective
    lat_lon_intp = lat_lon_intp.reshape((n_grid, n_grid))

    dept_time_intp = ( mu_zbar_effective + vbfpca.mu_theta_i_list_active[year_to_inspect][:,idx_beta_active_effective] ) @ phi_up_grid_depttime_effective
    dept_time_intp = dept_time_intp.reshape((n_grid, n_grid))
    et = time.time()
    logging.info(f'time for interpolation: {et-st:.2f}s')

    z_c = mu_zbar_effective.T @ mu_zbar_effective + vbfpca.cov_zbar_active[ np.ix_( idx_beta_active_effective, idx_beta_active_effective ) ]
    t_c = vbfpca.mu_theta_i_list_active[year_to_inspect][:,idx_beta_active_effective].T @ vbfpca.mu_theta_i_list_active[year_to_inspect][:,idx_beta_active_effective] + vbfpca.cov_theta_i_list_active[year_to_inspect][ np.ix_( idx_beta_active_effective, idx_beta_active_effective ) ]
    zt = z_c + t_c + vbfpca.mu_theta_i_list_active[year_to_inspect][:,idx_beta_active_effective].T @ mu_zbar_effective * 2

    lat_lon_unc =  np.trace( (phi_up_grid_latlon_effective.T[:,:,None] @ phi_up_grid_latlon_effective.T[:,None,:]) @ ( zt )[None,...], axis1=1, axis2=2 ).reshape((n_grid, n_grid)) + (vbfpca.b_white/ (vbfpca.a_white - 1) ) - lat_lon_intp**2

    dept_time_unc =  np.trace( (phi_up_grid_depttime_effective.T[:,:,None] @ phi_up_grid_depttime_effective.T[:,None,:]) @ ( zt )[None,...], axis1=1, axis2=2 ).reshape((n_grid, n_grid)) + (vbfpca.b_white/ (vbfpca.a_white - 1) ) - dept_time_intp**2


    # Call the function
    plot_interpolated_temperature(
        lat_lon_intp=lat_lon_intp,
        dept_time_intp=dept_time_intp,
        # train_points= np.hstack((z, temp)),
        train_points= np.hstack( ( x_up_list[year_to_inspect][0], y_up_list[year_to_inspect][0][:,None] * y_up_all_std + y_up_all_mean ) ),
        # test_points= np.array([dept, lat, lon, t, temp]),
        test_points= np.hstack( ( x_up_list_test[year_to_inspect][0], y_up_list_test[year_to_inspect][0][:,None] * y_up_all_std + y_up_all_mean ) ),
        ref_point= np.array([dept, lat, lon, t, temp]),
        
        y = year_list[year_to_inspect],
        n_grid=n_grid,
        path_to_save=f'{dir_debug}/interpolated_data_temperature_map_{year}dp{dept:.0f}lt{lat:.0f}ln{lon:.0f}tm{t:.2f}.pdf',
        depth_range=depth_range,

        threshold_depth = 1,
        threshold_time = 1/365,
        threshold_geodetic = 50,

    )
