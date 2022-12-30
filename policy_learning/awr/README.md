# Advantage-Weighted Regression (AWR)

This directory contains slightly modified version of the [AWR](https://github.com/xbpeng/awr).

## Training Models

To train a policy, run the following command:

``python ./scripts/run_script.py --env maze2d-medium-dense-v1 --max_iter 2000 --seed 0 --reward_path [reward_path]``

- `maze2d-open-dense-v0` can be replaced with other environments.
- `--max_iter` specifies the maximum number of training iterations.
- `--seed` specifies random seed
- `--reward_path` specifies the reward predictions generated during the reward learning phase. e.g. ~/reward_learning/rewards/ensemble_maze2d-medium-dense-v1_initial_pairs_10_num_queries_1_num_iter_5_retrain_num_iter_1_voi_dis_seed_0_round_num_0.npy
- The log and model will be saved to the `output/` directory by default. But the output directory can also be specified with `--output_dir [output-directory]`.

It may be necessary to add this directory to your PYTHONPATH before running.