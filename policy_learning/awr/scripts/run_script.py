import argparse
import gym
import uuid
import numpy as np
import json
import os
import sys
import tensorflow as tf

import d4rl

import awr_configs
import learning.awr_agent as awr_agent
import learning.random_agent as random_agent

arg_parser = None

def parse_args(args):
    parser = argparse.ArgumentParser(description="Train or test control policies.")

    parser.add_argument("--env", dest="env", default="")

    parser.add_argument("--train", dest="train", action="store_true", default=True)
    parser.add_argument("--test", dest="train", action="store_false", default=True)

    parser.add_argument("--max_iter", dest="max_iter", type=int, default=np.inf)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_episodes", dest="test_episodes", type=int, default=32)
    parser.add_argument("--output_dir", dest="output_dir", default="output")
    parser.add_argument("--output_iters", dest="output_iters", type=int, default=50)
    parser.add_argument("--model_file", dest="model_file", default="")

    parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
    parser.add_argument("--gpu", dest="gpu", default="")
    parser.add_argument("--mask_ratio", dest="mask_ratio", type=float, default=0.0)
    parser.add_argument("--mask_with_zero", dest="mask_with_zero", action="store_true", default=False)
    parser.add_argument("--mask_seed", type=int, default=0)
    parser.add_argument("--reward_path", dest="reward_path", default="")
    parser.add_argument("--random_policy", dest="random_policy", action="store_true", default=False)

    arg_parser = parser.parse_args()

    return arg_parser

def enable_gpus(gpu_str):
    if (gpu_str is not ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return

def build_env(env_id):
    assert(env_id is not ""), "Unspecified environment."
    env = gym.make(env_id)
    return env

def build_agent(env, arg_parser):
    env_id = arg_parser.env
    agent_configs = {
        'mask_ratio': arg_parser.mask_ratio,
        'mask_with_zero': arg_parser.mask_with_zero,
        'mask_seed': arg_parser.mask_seed,
        'reward_path': arg_parser.reward_path,
        'seed': arg_parser.seed, 
    }
    if (env_id in awr_configs.AWR_CONFIGS):
        agent_configs = awr_configs.AWR_CONFIGS[env_id]

    graph = tf.Graph()
    sess = tf.Session(graph=graph)

    if arg_parser.random_policy:
        agent = random_agent.RandomAgent(env=env, sess=sess, **agent_configs)
    else:
        agent = awr_agent.AWRAgent(env=env, sess=sess, **agent_configs)

    return agent

def main(args):
    d4rl.set_dataset_path('./datasets')

    global arg_parser
    arg_parser = parse_args(args)
    enable_gpus(arg_parser.gpu)

    # Setup logging
    if arg_parser.random_policy:
        if not arg_parser.reward_path:
            mid_output_dir = f'{arg_parser.env}_iter_{arg_parser.max_iter}_zero_{arg_parser.mask_with_zero}_ratio_{arg_parser.mask_ratio}_random_policy_seed_{arg_parser.mask_seed}'
        else:
            mid_output_dir = f'{arg_parser.env}_iter_{arg_parser.max_iter}_{arg_parser.reward_path}_random_policy_seed_{arg_parser.mask_seed}'
    else:
        if not arg_parser.reward_path:
            mid_output_dir = f'{arg_parser.env}_iter_{arg_parser.max_iter}_zero_{arg_parser.mask_with_zero}_ratio_{arg_parser.mask_ratio}_seed_{arg_parser.mask_seed}'
        else:
            mid_output_dir = f'{arg_parser.env}_iter_{arg_parser.max_iter}_{arg_parser.reward_path}_seed_{arg_parser.mask_seed}'

    final_output_dir = os.path.join(arg_parser.output_dir, mid_output_dir, str(uuid.uuid4()))
    os.makedirs(final_output_dir, exist_ok=True)
    with open(os.path.join(final_output_dir, 'params.json'), 'w') as params_file:
        json.dump({
            'env_name': arg_parser.env,
            'seed': arg_parser.seed,
        }, params_file)

    env = build_env(arg_parser.env)
    agent = build_agent(env, arg_parser)
    agent.visualize = arg_parser.visualize
    if (arg_parser.model_file is not ""):
        agent.load_model(arg_parser.model_file)


    if (arg_parser.train):
        agent.train(max_iter=arg_parser.max_iter,
                    test_episodes=arg_parser.test_episodes,
                    output_dir=final_output_dir,
                    output_iters=arg_parser.output_iters,
                    env_id = arg_parser.env
                    )
    else:
        agent.eval(num_episodes=arg_parser.test_episodes)

    return

if __name__ == "__main__":
    main(sys.argv)