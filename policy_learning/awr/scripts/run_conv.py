import argparse
import gym
import numpy as np
import os
import sys
import tensorflow as tf

import d4rl

import awr_configs
import learning.conv_awr_agent as awr_agent

arg_parser = None

def parse_args(args):
    parser = argparse.ArgumentParser(description="Train or test control policies.")

    parser.add_argument("--env", dest="env", default="carla-lane-v0")

    parser.add_argument("--train", dest="train", action="store_true", default=True)
    parser.add_argument("--test", dest="train", action="store_false", default=True)

    parser.add_argument("--max_iter", dest="max_iter", type=int, default=int(1e6))
    parser.add_argument("--test_episodes", dest="test_episodes", type=int, default=32)
    parser.add_argument("--output_dir", dest="output_dir", default="output")
    parser.add_argument("--output_iters", dest="output_iters", type=int, default=50)
    parser.add_argument("--model_file", dest="model_file", default="")

    parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
    parser.add_argument("--gpu", dest="gpu", default="")

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

def build_agent(env):
    env_id = arg_parser.env
    agent_configs = {}
    if (env_id in awr_configs.AWR_CONFIGS):
        agent_configs = awr_configs.AWR_CONFIGS[env_id]

    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    agent = awr_agent.AWRAgent(env=env, sess=sess, **agent_configs)

    return agent

def main(args):
    global arg_parser
    arg_parser = parse_args(args)
    enable_gpus(arg_parser.gpu)

    env = build_env(arg_parser.env)

    agent = build_agent(env)
    agent.visualize = arg_parser.visualize
    if (arg_parser.model_file is not ""):
        agent.load_model(arg_parser.model_file)


    if (arg_parser.train):
        agent.train(max_iter=arg_parser.max_iter,
                    test_episodes=arg_parser.test_episodes,
                    output_dir=arg_parser.output_dir,
                    output_iters=arg_parser.output_iters)
    else:
        agent.eval(num_episodes=arg_parser.test_episodes)

    return

if __name__ == "__main__":
    main(sys.argv)
