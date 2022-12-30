import argparse
import pickle
import gym
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import d4rl # Import required to register environments
import deepdish as dd
import os
from utils import *

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. maze2d-medium-dense-v1')
    parser.add_argument('--initial_pairs', default = 10, type=int, help="initial number of pairs of trajectories used to train the reward models")
    parser.add_argument('--num_snippets', default = 0, type = int, help = "number of short subtrajectories to sample")
    parser.add_argument('--voi', default='', help='Choose between infogain, disagreement, or random')
    parser.add_argument('--num_rounds', default = 0, type = int, help = "number of rounds of active querying")
    parser.add_argument('--num_queries', default = 1, type = int, help = "number of queries per round of active querying")
    parser.add_argument('--num_iter', default = 5, type = int, help = "number of iteration of initial data")
    parser.add_argument('--retrain_num_iter', default = 1, type = int, help = "number of training iteration after one round of active querying")
    parser.add_argument('--num_ensembles', default = 7, type = int, help = "number of ensemble of members")
    parser.add_argument('--seed', default = 0, type = int, help = "random seed")
    parser.add_argument('--beta', default = 10, type = int, help = "beta as a measure of confidence for info gain")

    args = parser.parse_args()

    # Torch RNG
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Python RNG
    np.random.seed(args.seed)

    env_name = args.env_name
    list_env_name = list(env_name.split("-"))
    maze_name = list_env_name[1]
    env = gym.make(args.env_name)

    env_prefix = env_name.split('-')[0]
    dataset = env.get_dataset()

    #set input_dim based on environment
    if env_prefix == 'maze2d':
        input_dim = 4
    elif env_prefix == 'halfcheetah':
        input_dim = 17
    elif env_prefix == 'hopper':
        input_dim = 11
    elif env_prefix == 'kitchen':
        input_dim = 60
    elif env_name == 'flow-ring-random-v1' or env_name == 'flow-ring-random-v0':
        input_dim = 4
    elif env_name == 'flow-merge-random-v1' or env_name == 'flow-merge-random-v0':
        input_dim = 60

    initial_pairs = args.initial_pairs
    num_snippets = args.num_snippets
    min_snippet_length = 25 #min length of trajectory for training comparison
    maximum_snippet_length = 100

    traj_length = 50
    num_iter = args.num_iter
    retrain_num_iter = args.retrain_num_iter
    #arg
    num_queries = args.num_queries
    voi = args.voi
    num_rounds = args.num_rounds

    num_seeds = args.num_ensembles

    beta = args.beta

    #check if a directory exists
    path = "./rewards"
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)

    reward_model_path = os.path.join(path, f'./ensemble_{env_name}_initial_pairs_{initial_pairs}_num_queries_{num_queries}_num_iter_{num_iter}_retrain_num_iter_{retrain_num_iter}_voi_{voi}_seed_{args.seed}')
    active_reward_root = os.path.join(path, f'./ensemble_{env_name}_initial_pairs_{initial_pairs}_num_queries_{num_queries}_num_iter_{num_iter}_retrain_num_iter_{retrain_num_iter}_voi_{voi}_seed_{args.seed}_round_num_')

    lr = 0.00005
    weight_decay = 0.0
    num_iter = args.num_iter #num times through training data
    l1_reg=0.0
    stochastic = True

    demo_list = []
    returns_list = [] 
    rewards_list = []
    models_list = []
    training_obs_list, training_labels_list = [], []

    #pretrain the reward models with intial number of query pairs
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        demonstrations, learning_returns, learning_rewards = generate_novice_demos(dataset, initial_pairs, traj_length)
        demo_list.append(demonstrations)
        returns_list.append(learning_returns)
        rewards_list.append(learning_rewards)

        #sort the demonstrations according to ground truth reward to simulate ranked demos
        demo_lengths = [len(d) for d in demonstrations]
        max_snippet_length = min(np.min(demo_lengths), maximum_snippet_length)
        demonstrations = [x for _, x in sorted(zip(learning_returns,demonstrations), key=lambda pair: pair[0])]

        sorted_returns = sorted(learning_returns)

        training_obs, training_labels = create_training_data(demonstrations, initial_pairs, num_snippets, min_snippet_length, max_snippet_length)

        training_obs_list.append(training_obs)
        training_labels_list.append(training_labels)

        # Now we create a reward Network and optimize it using the training data.
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        reward_net = Net(input_dim)
        reward_net.to(device)
        import torch.optim as optim
        optimizer = optim.Adam(reward_net.parameters(),  lr=lr, weight_decay=weight_decay)
        learn_reward(reward_net, optimizer, training_obs, training_labels, num_iter, l1_reg, reward_model_path)

        models_list.append(reward_net)
    
    #just save the first model
    reward_net = models_list[0]

    #load a separate demonstrations that contains a lot or all of the trajectories, randomly sample a bunch, demos returns, rewards
    large_num_trajs =  int(dataset['observations'].shape[0] / traj_length) // 10
    large_num_pairs = large_num_trajs * 5
    large_demonstrations, large_learning_returns, large_learning_rewards = generate_novice_demos(dataset, large_num_trajs, traj_length)

    #sort the demonstrations according to ground truth reward to simulate ranked demos
    large_demo_lengths = [len(d) for d in large_demonstrations]
    large_max_snippet_length = min(np.min(large_demo_lengths), maximum_snippet_length)
    sorted_large_demonstrations = [x for _, x in sorted(zip(large_learning_returns,large_demonstrations), key=lambda pair: pair[0])]

    large_sorted_returns = sorted(large_learning_returns)

    large_training_obs, large_training_labels = create_training_data(sorted_large_demonstrations, large_num_pairs, num_snippets, min_snippet_length, max_snippet_length)

    #for calculating reward npy
    npy_traj_length =  1000
    npy_num_trajs =  int(dataset['observations'].shape[0] / npy_traj_length)
    npy_demonstrations, _, _ = generate_novice_demos(dataset, npy_num_trajs, npy_traj_length)

    acc = calc_accuracy(models_list[0], large_training_obs, large_training_labels)

    #tail case for hopper-medium-expert which has length 1999906
    num_tails = dataset['observations'].shape[0] % npy_traj_length
    step_start = dataset['observations'].shape[0] - num_tails #for generate_novice_demos
    npy_demonstrations_tail, _, _ = generate_novice_demos(dataset, num_tails, 1, steps=step_start)

    reward_arr_list = []
    for model_idx in range(len(models_list)):
        reward_net = models_list[model_idx]
        reward_arr = np.array(parallel_predict_reward_sequence(reward_net, npy_demonstrations))
        reward_arr_tail = np.array(parallel_predict_reward_sequence(reward_net, npy_demonstrations_tail))
        reward_arr_comb = np.concatenate((reward_arr, reward_arr_tail), axis=0)
        reward_arr_list.append(reward_arr_comb)

    reward_arr_all = np.array(reward_arr_list)
    reward_arr = np.mean(reward_arr_all, axis=0)
    with open(active_reward_root+str(0)+'.npy', 'wb') as f:
        np.save(f, reward_arr)
    
    #number of times we query and retrain
    for round in range(num_rounds):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loss_criterion = nn.CrossEntropyLoss()
        num_correct = 0.

        var_list = []
        infogain_list = []
        with torch.no_grad():
            for i in range(len(large_training_obs)):
                label = large_training_labels[i]
                traj_i, traj_j = large_training_obs[i]
                traj_i = np.array(traj_i)
                traj_j = np.array(traj_j)
                traj_i = torch.from_numpy(traj_i).float().to(device)
                traj_j = torch.from_numpy(traj_j).float().to(device)

                #disagreement variant
                pred_label_list = []
                num_ones = 0.
                for i in range(len(models_list)):
                    reward_network = models_list[i]
                    #forward to get logits
                    outputs, abs_return = reward_network.forward(traj_i, traj_j)
                    _, pred_label = torch.max(outputs,0)
                    pred_label_list.append(pred_label.item())
                    if pred_label.item() == 1:
                        num_ones += 1.

                p = num_ones / len(models_list)
                var = p * (1. - p)
                var_list.append(var)

                #infogain variant
                rewards_i = []
                rewards_j = []

                num_ones = 0.
                for i in range(len(models_list)):
                    reward_network = models_list[i]
                    #forward to get logits
                    cum_r_i, cum_r_j = reward_network.get_returns(traj_i, traj_j)
                    rewards_i.append(cum_r_i.item())
                    rewards_j.append(cum_r_j.item())

                infogain = pref_info_gain(rewards_i, rewards_j, beta=beta)
                infogain_list.append(infogain)

            avg_var = sum(var_list) / len(var_list)
            avg_infogain = sum(infogain_list) / len(infogain_list)
            if voi == 'dis':
                var_arr = np.array(var_list)
                query_idx = var_arr.argsort()[-num_queries:][::-1]
            elif voi == 'info':
                infogain_arr = np.array(infogain_list)
                query_idx = infogain_arr.argsort()[-num_queries:][::-1]
            else:
                # random index
                l = [i for i in range(len(large_training_obs))]
                query_idx = random.sample(l, num_queries)

            for i in range(len(training_obs_list)):
                for idx in query_idx:
                    training_obs_list[i].append(large_training_obs[idx])
                    training_labels_list[i].append(large_training_labels[idx])
        
        #retrain the reward models
        for i in range(num_seeds):
            torch.manual_seed(i)
            np.random.seed(i)

            training_obs, training_labels = training_obs_list[i], training_labels_list[i]

            # Now we create a reward network and optimize it using the training data.
            reward_net = models_list[i]
            import torch.optim as optim
            optimizer = optim.Adam(reward_net.parameters(),  lr=lr, weight_decay=weight_decay)
            learn_reward(reward_net, optimizer, training_obs, training_labels, retrain_num_iter, l1_reg, reward_model_path)

        #just save the first model
        reward_net = models_list[0]
        round_idx = round + 1

        acc = calc_accuracy(models_list[0], large_training_obs, large_training_labels)

        reward_arr_list = []
        for model_idx in range(len(models_list)):
            reward_net = models_list[model_idx]
            reward_arr = np.array(parallel_predict_reward_sequence(reward_net, npy_demonstrations))
            reward_arr_tail = np.array(parallel_predict_reward_sequence(reward_net, npy_demonstrations_tail))
            reward_arr_comb = np.concatenate((reward_arr, reward_arr_tail), axis=0)
            reward_arr_list.append(reward_arr_comb)
        reward_arr_all = np.array(reward_arr_list)
        reward_arr = np.mean(reward_arr_all, axis=0)
        with open(active_reward_root+str(round+1)+'.npy', 'wb') as f:
            np.save(f, reward_arr)