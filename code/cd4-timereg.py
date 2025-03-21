# %%
import os
import numpy as np
import argparse
import pprint
from util import get_kernel_list_and_coef_func_index
from bsfda import BSFDA

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
    datefmt='%Y-%m-%d %A %H:%M:%S')


np.random.seed(31415926)
# %%
parser = argparse.ArgumentParser()
parser.add_argument('--dir_debug', type=str)
parser.add_argument('--n_cluster_kernel', type=int, default=5)
parser.add_argument('--log_transform', default=False, help='whether to apply the log transform to the original data. https://rdrr.io/rforge/ALA/man/cd4.html#:~:text=a%20numeric%20vector%3B%20log%20transformed%20CD4%20counts%20(log(CD4%20counts%20%2B%201))', action = 'store_true')

args = parser.parse_args()
logging.critical('args is \n%s'%(pprint.pformat(args.__dict__, indent=4)))

dir_debug = args.dir_debug
n_cluster_kernel = args.n_cluster_kernel


# %%
matrix = np.loadtxt(
    # data is from https://rdrr.io/cran/timereg/man/cd4.html
    fname='./data/cd4-tiemreg/cd4-timereg.csv',
    dtype= str,
    delimiter= ','
)
id2z_y_list = {}
for row in matrix[1:]:
    i = int(row[1].replace('"',''))
    zy = ( row[-2].astype(float), row[-4].astype(float) )
    if i in id2z_y_list:
        id2z_y_list[i].append(zy)
    else:
        zy0 = ( 0, row[-5].astype(float) )
        id2z_y_list[i] = [zy0,zy]
id2z_y_list = sorted(list(id2z_y_list.items()))

x_up_list = [
    np.array([
        i2[0] for i2 in id_zy[1]
    ])[None,:]
    for i, id_zy in enumerate(id2z_y_list)
]
y_up_list = [
    np.array([
        i2[1] for i2 in id_zy[1]
    ] )[None,:]
    for i, id_zy in enumerate(id2z_y_list)
]
if args.log_transform:
    y_up_list = [
        np.log(y+1)
        for y in y_up_list
    ]

# %%
if not os.path.exists(dir_debug):
    os.makedirs(dir_debug)

z_all = matrix[1:,-2].astype(float)

kernel_list, coef_func_index = get_kernel_list_and_coef_func_index(
    x_up_list= x_up_list,
    y_up_list= y_up_list,
    n_cluster_kernel= n_cluster_kernel,
    dir_debug= dir_debug,
    normalize_rbf= True,
    kernel_coef = 3.0,
)

# %%
vbfpca = BSFDA(kernel_list= kernel_list).fit(
    y_up_list= y_up_list,
    x_up_list= [x[...,None] for x in x_up_list],

    coef_func_index = coef_func_index,

    n_log=100,
    dir_debug = dir_debug,
)
vbfpca.plot_results(
    path_fig=f'{dir_debug}/bfpca_truth_vs_predict.pdf',
)
vbfpca.dump(path_dump=f'{dir_debug}/vbfpca.pkl')