# yehua, n_job is the number of jobs to run in parallel
python -u ./code/yehua.py --snro 1 --m 5 --n_ck 10 --n_job 200 --dir_out ./data/yehua/s1/m5/out
python -u ./code/yehua.py --snro 1 --m 10 --n_ck 10 --n_job 200 --dir_out ./data/yehua/s1/m10/out
python -u ./code/yehua.py --snro 1 --m 50 --n_ck 10 --n_job 200 --dir_out ./data/yehua/s1/m50/out

python -u ./code/yehua.py --snro 2 --m 5 --n_ck 10 --n_job 200 --dir_out ./data/yehua/s2/m5/out
python -u ./code/yehua.py --snro 2 --m 10 --n_ck 10 --n_job 200 --dir_out ./data/yehua/s2/m10/out
python -u ./code/yehua.py --snro 2 --m 50 --n_ck 10 --n_job 200 --dir_out ./data/yehua/s2/m50/out

python -u ./code/yehua.py --snro 3 --m 5 --n_ck 10 --n_job 200 --dir_out ./data/yehua/s3/m5/out
python -u ./code/yehua.py --snro 3 --m 10 --n_ck 10 --n_job 200 --dir_out ./data/yehua/s3/m10/out
python -u ./code/yehua.py --snro 3 --m 50 --n_ck 10 --n_job 200 --dir_out ./data/yehua/s3/m50/out

python -u ./code/yehua.py --snro 4 --m 5 --n_ck 10 --n_job 200 --dir_out ./data/yehua/s4/m5/out
python -u ./code/yehua.py --snro 4 --m 10 --n_ck 10 --n_job 200 --dir_out ./data/yehua/s4/m10/out
python -u ./code/yehua.py --snro 4 --m 50 --n_ck 10 --n_job 200 --dir_out ./data/yehua/s4/m50/out

python -u ./code/yehua.py --snro 5 --m 5 --n_ck 5 --n_job 200 --dir_out ./data/yehua/s5/m5/out
python -u ./code/yehua.py --snro 5 --m 10 --n_ck 10 --n_job 200 --dir_out ./data/yehua/s5/m10/out
python -u ./code/yehua.py --snro 5 --m 50 --n_ck 10 --n_job 200 --dir_out ./data/yehua/s5/m50/out

# 4d simulation
python -u ./code/ndindex.py --n 200 --m 50 --ls 0.33 --d_index_set 4 --n_log 100 --dir_out ./data/ndindex/out

# cd4
python -u ./code/cd4-timereg.py --dir_debug ./data/cd4-timereg/out

# wind
python -u ./code/wind.py --dir_debug ./data/wind/out

# argo
python -u ./code/argo.py

# gibbs vs mean-field
python -u ./code/gibbs_vs_mf.py