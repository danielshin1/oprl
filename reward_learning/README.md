# Running Reward Learning
To run reward learning, run the following command:
```
python TREX_ensemble.py --voi dis --env_name maze2d-medium-dense-v1 --initial_pairs 10 --num_rounds 10 --num_queries 1
```
- Use `TREX_dropout.py` instead of `TREX_ensemble.py` to use dropout instead of ensemble of models to represent uncertainty.
- `--voi` specifies the variant of estimated value of information used for active query selection(dis = disagreement, info= information gain). If left empty, uses random queries. 
- `maze2d-medium-dense-v1` can be replaced with other environments.
- `--initial_pairs` specifies the initial number of pairs of trajectories used to train the reward models.
- `--num_rounds` specifies the number of rounds of querying for additional pairs of trajectories and labels.
- `--num_queries` specifies the number of query pairs per round.
- The model predictions will be saved to the `rewards/` directory by default.