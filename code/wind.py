# %%
import os
import numpy as np
import datetime
import pandas as pd
from util import get_kernel_list_and_coef_func_index
from bsfda import BSFDA

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
    datefmt='%Y-%m-%d %A %H:%M:%S')


np.random.seed(31415926)

# %%
use_wind_speed = True

# %%
date_format = '%Y-%m-%dT%H:%M:%SZ'
date_time_str0 = '2022-07-15T00:00:00Z'
date_time_obj0 = datetime.datetime.strptime(date_time_str0, date_format)
parse_time = lambda date_str, date_time_obj0=date_time_obj0: ( datetime.datetime.strptime(date_str, date_format) - date_time_obj0 ).total_seconds()

# %%
dir_root = './data/wind'
matrix = pd.read_csv(f'{dir_root}/SLC_wind_out_July.csv').to_numpy()

z_all = []
id2z_y_list = {}
for row in matrix[1:]:
    i = (row[3].replace('"',''))
    z = parse_time(row[-3])
    z_all.append(z)
    y = row[-2] if use_wind_speed else row[-1]
    if i in id2z_y_list:
        id2z_y_list[i].append([z,y])
    else:
        id2z_y_list[i] = [[z,y]]
id2z_y_list = sorted(list(id2z_y_list.items()))
logging.info(f'len(id2z_y_list)={len(id2z_y_list)}')
id2z_y_list = [e for e in id2z_y_list if len(e[1])>=5]
logging.info(f'len(id2z_y_list)={len(id2z_y_list)}')

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

# %%
dir_debug= f'{dir_root}/out-speed_kc4' if use_wind_speed else f'{dir_root}/out-direction_kc4'
n_cluster_kernel= 10

if not os.path.exists(dir_debug):
    os.makedirs(dir_debug)


z_all=np.array(z_all)

kernel_list, coef_func_index = get_kernel_list_and_coef_func_index(
    x_up_list= x_up_list,
    y_up_list= y_up_list,
    n_cluster_kernel= n_cluster_kernel,
    dir_debug= dir_debug,
    normalize_rbf= True,
    kernel_coef= 1e4,
)


# %%
vbfpca = BSFDA(kernel_list= kernel_list).fit(
    y_up_list= y_up_list,
    x_up_list= [x[...,None] for x in x_up_list],

    coef_func_index=coef_func_index,

    n_log=100,
    dir_debug = dir_debug,
)
vbfpca.plot_results(
    path_fig=f'{dir_debug}/bfpca_truth_vs_predict.pdf',
)
vbfpca.dump(path_dump=f'{dir_debug}/vbfpca.pkl')
# %%
