# %%
import os
import numpy as np
import logging

logging.basicConfig(
    level= logging.INFO,
    format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
    datefmt='%Y-%m-%d %A %H:%M:%S')

np.random.seed( 0 )

from util import get_kernel_list_and_coef_func_index
from bsfda import BSFDA

logging.critical(f'This is an example of BSFDA (Bayesian Scalable Functional Data Analysis). The data is generated from the same model as scenario 1 with medium sampling in the paper of yehua "Selecting the Number of Principal Components in Functional Data". The number of sample functions is decreased from 200 to 30 for a quick demo, which takes around 10 minutes to run on a laptop. The results include the number of components, the number of basis functions and the noise level, with more details saved in the directory "./example.py_debug".')

# %%
'''
generate example data
'''

# number of sample functions
n = 20

# number of observations per function, which can take a list of integers (randomly choose one for each sample function)
m = [3] + [10]*4

# mean function
mu = lambda t: 5 * (t-0.6)**2

# eigen values
w = np.array([0.6, 0.3, 0.1])

# number of components
p0 = w.shape[0]

# variance of the white noise
sigma_u_sq = 0.2
sigma_u = sigma_u_sq**(0.5)

logging.critical(f'Standard deviation of the white noise is: {sigma_u:0.3f}')
logging.critical(f'Number of components is: {p0}')

# eigen functions
phi = lambda t : np.stack([
    np.ones_like(t), 
    2**0.5 * np.sin(2*np.pi*t), 
    2**0.5 * np.cos(2*np.pi*t),
    ], axis=0)

# low dimensional latent variables
ksi = np.random.randn( n, p0 ) @ np.diag( w**0.5 )


w_up_true= np.diag( w**0.5 )

# obersevation index set
t_list = [
    np.sort(np.random.random(size=( m[int(np.floor(np.random.rand(1)*(len(m))))] )))[None,:] for i in range(n)
]
phi_up_list = [
    phi(t_list[i].flatten()) for i in range(n)
]

# noiseless observations
x_up_list = [
    ksi[i,...] @ phi_up_list[i] + mu( t_list[i] ) 
    for i,ti in enumerate(t_list)
]
# additive white noise
u_up_list = [
    sigma_u * np.random.randn(1, ti.shape[1]) for i,ti in enumerate(t_list)
]
# noisy observations
w_up_list = [
    x_up_list[i] + u_up_list[i] for i,ti in enumerate(t_list)
]


# %%
# directory to save the debug information
dir_debug = os.path.abspath('./data/example.py_debug')

if not os.path.exists(dir_debug):
    os.makedirs(dir_debug)
    

# %%
# get the kernel list using cross validation
kernel_list = get_kernel_list_and_coef_func_index(
    x_up_list= t_list,
    y_up_list= w_up_list,
    dir_debug= dir_debug,
    )[0]


# %%
# fit the model
vbfpca = BSFDA(
    kernel_list= kernel_list,
)
# it will converge after around 5000 iterations
vbfpca = vbfpca.fit(
    y_up_list= w_up_list,
    x_up_list= [t[...,None] for t in t_list],
    # number of iterations to log
    n_log= 500,
    dir_debug = dir_debug,
)
vbfpca.dump(path_dump=f'{dir_debug}/vbfpca.pkl', exclude_data=True)


# %%
# print the results
logging.critical(f'Here are the estimated parameters:')
logging.critical(f'Standard deviation of the white noise is: {((vbfpca.a_white / vbfpca.b_white)**(-0.5)):0.3f}')
logging.critical(f'Number of components is {len(vbfpca.get_idx_alpha_active_effective())}')
logging.critical(f'Number of basis functions is {len(vbfpca.get_idx_beta_active_effective())}')
logging.critical(f'More detailed results are saved in {dir_debug}')


# # %%
# plot the results vs the truth
d0_index_set_plot = 0

mu_mise, cov_mise = vbfpca.plot_results(
    d0_index_set_plot = d0_index_set_plot,

    sigma_true= sigma_u,
    w_up_true= w_up_true,
    mean_true = lambda x: mu(x)[...,d0_index_set_plot],
    w_x_phi_up_list_true= lambda x: w_up_true @ phi( x )[...,d0_index_set_plot],
    
    path_fig=f'{dir_debug}/bfpca_truth_vs_predict.pdf',
)
i_list = [0,18]
mu_mise, cov_mise = vbfpca.plot_results(
    d0_index_set_plot = d0_index_set_plot,

    sigma_true= sigma_u,
    w_up_true= w_up_true,
    mean_true = lambda x: mu(x)[...,d0_index_set_plot],
    w_x_phi_up_list_true= lambda x: w_up_true @ phi( x )[...,d0_index_set_plot],
    
    path_fig=f'{dir_debug}/bfpca_truth_vs_predict_i{i_list}.pdf',
    n_max_uncertainty= 2,
    i_list= i_list,
)


# %%
