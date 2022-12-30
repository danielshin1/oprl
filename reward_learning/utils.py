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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parallel_predict_reward_sequence(net, traj):
    rewards_from_obs = []
    steps = 0
    batch_size = 10
    with torch.no_grad():
        for s in traj:
            steps += 1
            if steps % 100000 == 0:
                print(f'steps: {steps}')
            r = net.parallel_cum_return(torch.from_numpy(np.array([s])).float().to(device))[0].tolist()
            if isinstance(r,float):
                rewards_from_obs.append(r)
            else:
                rewards_from_obs.extend(r)
    return rewards_from_obs

def generate_novice_demos(dataset, num_trajs, traj_length, steps=0):
    demonstrations = []
    learning_returns = []
    learning_rewards = []

    for i in range(num_trajs):
        done = False
        traj = []
        gt_rewards = []
        r = 0

        acc_reward = 0

        while True:
            ob, r, done = dataset['observations'][steps], dataset['rewards'][steps], dataset['terminals'][steps]
            traj.append(ob)
            gt_rewards.append(r)
            steps += 1
            acc_reward += r
            # no terminal states for maze
            # if done:
            if steps % traj_length == 0:
                # print("checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps,acc_reward))
                # print("steps: {}, return: {}".format(steps,acc_reward))
                break
        demonstrations.append(traj)
        learning_returns.append(acc_reward)
        learning_rewards.append(gt_rewards)

    return demonstrations, learning_returns, learning_rewards

def create_training_data(demonstrations, num_trajs, num_snippets, min_snippet_length, max_snippet_length):
    #collect training data
    max_traj_length = 0
    training_obs = []
    training_labels = []
    num_demos = len(demonstrations)

    #add full trajs (for use on Enduro)
    for n in range(num_trajs):
        ti = 0
        tj = 0
        #only add trajectories that are different returns
        while(ti == tj):
            #pick two random demonstrations
            ti = np.random.randint(num_demos)
            tj = np.random.randint(num_demos)
        #create random partial trajs by finding random start frame and random skip frame
        # si = np.random.randint(6)
        # sj = np.random.randint(6)
        # step = np.random.randint(3,7)
        
        traj_i = demonstrations[ti]  #slice(start,stop,step)
        traj_j = demonstrations[tj]
        
        if ti > tj:
            label = 0
        else:
            label = 1
        
        training_obs.append((traj_i, traj_j))
        training_labels.append(label)
        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))


    #fixed size snippets with progress prior
    for n in range(num_snippets):
        ti = 0
        tj = 0
        #only add trajectories that are different returns
        while(ti == tj):
            #pick two random demonstrations
            ti = np.random.randint(num_demos)
            tj = np.random.randint(num_demos)
        #create random snippets
        #find min length of both demos to ensure we can pick a demo no earlier than that chosen in worse preferred demo
        min_length = min(len(demonstrations[ti]), len(demonstrations[tj]))
        rand_length = np.random.randint(min_snippet_length, max_snippet_length)
        if ti < tj: #pick tj snippet to be later than ti
            ti_start = np.random.randint(min_length - rand_length + 1)
            #print(ti_start, len(demonstrations[tj]))
            tj_start = np.random.randint(ti_start, len(demonstrations[tj]) - rand_length + 1)
        else: #ti is better so pick later snippet in ti
            tj_start = np.random.randint(min_length - rand_length + 1)
            #print(tj_start, len(demonstrations[ti]))
            ti_start = np.random.randint(tj_start, len(demonstrations[ti]) - rand_length + 1)
        traj_i = demonstrations[ti][ti_start:ti_start+rand_length:2] #skip everyother framestack to reduce size
        traj_j = demonstrations[tj][tj_start:tj_start+rand_length:2]

        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))
        if ti > tj:
            label = 0
        else:
            label = 1
        training_obs.append((traj_i, traj_j))
        training_labels.append(label)

    return training_obs, training_labels

# Train the network
def learn_reward(reward_network, optimizer, training_inputs, training_outputs, num_iter, l1_reg, checkpoint_dir):
    #check if gpu available
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    # print(device)
    loss_criterion = nn.CrossEntropyLoss()
    
    cum_loss = 0.0
    training_data = list(zip(training_inputs, training_outputs))
    for epoch in range(num_iter):
        np.random.shuffle(training_data)
        training_obs, training_labels = zip(*training_data)
        for i in range(len(training_labels)):
            traj_i, traj_j = training_obs[i]
            labels = np.array([training_labels[i]])
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)
            labels = torch.from_numpy(labels).to(device)

            #zero out gradient
            optimizer.zero_grad()

            #forward + backward + optimize
            outputs, abs_rewards = reward_network.forward(traj_i, traj_j)
            outputs = outputs.unsqueeze(0)
            loss = loss_criterion(outputs, labels) + l1_reg * abs_rewards
            loss.backward()
            optimizer.step()

            #print stats to see if learning
            item_loss = loss.item()
            cum_loss += item_loss
            if i % 100 == 99:
                #print(i)
                # print("epoch {}:{} loss {}".format(epoch,i, cum_loss))
                # print(abs_rewards)
                cum_loss = 0.0
                # print("check pointing")
                # torch.save(reward_net.state_dict(), checkpoint_dir)
    # print("finished training")

def calc_accuracy(reward_network, training_inputs, training_outputs):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_criterion = nn.CrossEntropyLoss()
    num_correct = 0.
    with torch.no_grad():
        for i in range(len(training_inputs)):
            label = training_outputs[i]
            traj_i, traj_j = training_inputs[i]
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)

            #forward to get logits
            outputs, abs_return = reward_network.forward(traj_i, traj_j)
            _, pred_label = torch.max(outputs,0)
            if pred_label.item() == label:
                num_correct += 1.
    return num_correct / len(training_inputs)

def test_calc_accuracy(reward_network, training_inputs, training_outputs):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_criterion = nn.CrossEntropyLoss()
    num_correct = 0.
    with torch.no_grad():
        for i in range(len(training_inputs)):
            label = training_outputs[i]
            traj_i, traj_j = training_inputs[i]
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)

            #forward to get logits
            outputs, abs_return = reward_network.test_forward(traj_i, traj_j)
            _, pred_label = torch.max(outputs,0)
            if pred_label.item() == label:
                num_correct += 1.
    return num_correct / len(training_inputs)

def predict_reward_sequence(net, traj):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rewards_from_obs = []
    steps = 0
    with torch.no_grad():
        for s in traj:
            steps += 1
            if steps % 100000 == 0:
                print(f'steps: {steps}')
            r = net.cum_return(torch.from_numpy(np.array([s])).float().to(device))[0].item()
            rewards_from_obs.append(r)
    return rewards_from_obs

def predict_traj_return(net, traj):
    return sum(predict_reward_sequence(net, traj))


def stable_softmax(x, beta):
    #Bradley-Terry Luce-Shepherd function for prefs (essentially cross entropy loss)
    x = beta * np.array(x)
    z = x - np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    softmax = numerator / denominator
    return softmax

def gen_dropout_distribution_pair(traj_a, traj_b, dropout_net, num_dropout_samples, device):
    '''
        input: two trajs, a trained dropout network, the number of dropout samples, and the device (cpu or cuda)
        output: the posterior over the rewards for the two trajs
    '''
    # dropout_returns_a = np.zeros(num_dropout_samples)
    # dropout_returns_b = np.zeros(num_dropout_samples)
    dropout_returns_a = []
    dropout_returns_b = []
    dropout_masks = []
    #run the traj through the network
    #convert to pytorch tensor
    traj_a = np.array(traj_a)
    traj_a = torch.from_numpy(traj_a).float().to(device) 
    traj_b = np.array(traj_b)
    traj_b = torch.from_numpy(traj_b).float().to(device) 
    
    #first run traj_a through dropout net and save the masks for traj_b
    for _ in range(num_dropout_samples):
        cum_ret, _, dmask = dropout_net.cum_return(traj_a)
        dropout_masks.append(dmask)
        dropout_returns_a.append(cum_ret.item())
        
    for i in range(num_dropout_samples):
        cum_ret, _, _ = dropout_net.cum_return(traj_b, dropout_masks[i])
        # dropout_masks.append(dmask) #where is this dmask coming from
        dropout_returns_b.append(cum_ret.item())

    return dropout_returns_a, dropout_returns_b

def pref_info_gain(rewards_x, rewards_y, beta=1):
    '''
        input: rewards_x and rewards_y are lists of the different possible outcomes of two experiments involving x and y (x and y will typically be trajectories)
                These scores come from generating n samples of the posterior distribution
        output: the empirical estiamte of the expected information gain of running a preference query experiment between x and y on the parameters of the reward (score) model posterior
        I(theta; Pref | Data) = H(Pref | Data) - Expectation over thetas [H(Pref | theta, Data)]
        where theta is the random variable representing the true reward function params, Data is the set of prefs so far,
        and Pref is a possible pair of trejctories we ask the user for a preference over
        
        Note that beta is set to 1 by default. If you know the person provides noisy labels it could be set lower
        if you know the labels are always correct you could set them higher, but I think beta = 1 is a decent value
    '''
    
    assert(len(rewards_x) == len(rewards_y))
    
    #calculate the probability of a user preference given each pair of samples rewards
    rewards_xy = np.stack((rewards_x, rewards_y), axis=1)
    probs = stable_softmax(rewards_xy, beta)
    
    #compute the entropy
    hs = -np.sum(probs * np.log2(probs), axis=1)
    
    #compute average probs for each option
    ave_ps = np.mean(probs, axis=0)
    
    #compute the entropy of preference given the current data: H(Pref | Data)
    H1 = -np.sum(ave_ps * np.log2(ave_ps))
    
    #compute the expected conditional entropy given the outcome of the preference query: Expectation over thetas [H(Pref | theta, Data)]
    H2 = np.mean(hs)
    
    return H1 - H2 
    #this term will be large if H1 is high, i.e. if the model is quite uncertain (i.e. ave probs are close to 0.5)
    #and H2 is low, i.e., the individual samples from the posterior are quite certain 

#pseudo-code for active query selection using info-gain
def select_active_query(pref_pair_pool, dropout_net, num_dropout_samples, device, beta):
    '''
        input: pref_pair_pool is a list of unlabelled 2-tuples of possible trajectories to ask for a preference over
                dropout_net is a T-REX net with dropout that has been trained on the prefs collected so far
                num_dropout_samples is the number of times to run through the dropout_net to approximate the posterior
                device is either cpu or cuda to make sure data is loaded correctly for running through the nnet

        output: the pair from the pref_pair_pool with highest info gain and the corresponding info_gain
    '''
    # best_pair = None
    max_info_gain = -np.inf
    best_idx = None
    # for traj_a, traj_b in pref_pair_pool:
    info_gain_list = []
    for i in range(len(pref_pair_pool)):
        traj_a, traj_b = pref_pair_pool[i]
        rewards_a, rewards_b = gen_dropout_distribution_pair(traj_a, traj_b, dropout_net, num_dropout_samples, device)
        # print('rewards_a', rewards_a)
        # print('rewards_b', rewards_b)
        info_gain = pref_info_gain(rewards_a, rewards_b, beta)
        # print('info_gain', info_gain)
        info_gain_list.append(info_gain)

        # #check if best found so far
        # if info_gain > max_info_gain:
        #     max_info_gain = info_gain
        #     # best_pair = (traj_a, traj_b)
        #     best_idx = i

    return info_gain_list

def gen_dropout_distribution_pair(traj_a, traj_b, dropout_net, num_dropout_samples, device):
    '''
        input: two trajs, a trained dropout network, the number of dropout samples, and the device (cpu or cuda)
        output: the posterior over the rewards for the two trajs
    '''
    # dropout_returns_a = np.zeros(num_dropout_samples)
    # dropout_returns_b = np.zeros(num_dropout_samples)
    dropout_returns_a = []
    dropout_returns_b = []
    dropout_masks = []
    #run the traj through the network
    #convert to pytorch tensor
    traj_a = np.array(traj_a)
    traj_a = torch.from_numpy(traj_a).float().to(device) 
    traj_b = np.array(traj_b)
    traj_b = torch.from_numpy(traj_b).float().to(device) 
    
    #first run traj_a through dropout net and save the masks for traj_b
    for _ in range(num_dropout_samples):
        cum_ret, _, dmask = dropout_net.cum_return(traj_a)
        dropout_masks.append(dmask)
        dropout_returns_a.append(cum_ret.item())
        
    for i in range(num_dropout_samples):
        cum_ret, _, _ = dropout_net.cum_return(traj_b, dropout_masks[i])
        # dropout_masks.append(dmask) #where is this dmask coming from
        dropout_returns_b.append(cum_ret.item())

    return dropout_returns_a, dropout_returns_b


class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

        # define a simple MLP neural net
        self.net = []
        nin, nout = self.input_dim, 1
        hidden_sizes = [512, 256, 128, 64, 32]
        hs = [nin] + hidden_sizes + [nout]
        for h0,h1 in zip(hs, hs[1:]):
            self.net.extend([
                    nn.Linear(h0, h1),
                    nn.ReLU(),
                ])
        self.net.pop() # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)

    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
   
        #compute forward pass of reward network (we parallelize across frames so batch size is length of partial trajectory)
        x = traj
        x = x.view(-1,self.input_dim)
        r = self.net(x)

        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))
        return sum_rewards, sum_abs_rewards

    def parallel_cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0

        #compute forward pass of reward network (we parallelize across frames so batch size is length of partial trajectory)
        x = traj
        x = x.view(-1,self.input_dim)
        r = self.net(x)

        sum_rewards = r.squeeze()
        sum_abs_rewards = torch.abs(r).squeeze()
        return sum_rewards, sum_abs_rewards

    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j

    def get_returns(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)

        return cum_r_i, cum_r_j

class DropoutNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        # self.input_dim = 17
        # self.input_dim = 2

        # number of hidden nodes in each layer (512)
        self.hidden_sizes = [512, 256, 128, 64, 32]

        # hidden_sizes = [256, 256, 256, 256, 256, 128, 64, 32] # for walker2d

        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(self.input_dim, self.hidden_sizes[0])
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(self.hidden_sizes[0],self.hidden_sizes[1])
        self.fc3 = nn.Linear(self.hidden_sizes[1],self.hidden_sizes[2])
        self.fc4 = nn.Linear(self.hidden_sizes[2],self.hidden_sizes[3])
        # linear layer (n_hidden -> 10)
        self.fc5 = nn.Linear(self.hidden_sizes[3],self.hidden_sizes[4])
        self.fc6 = nn.Linear(self.hidden_sizes[4],1)

    def cum_return(self, traj, mask=None):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        #compute forward pass of reward network (we parallelize across frames so batch size is length of partial trajectory)

        #if mask is given then use it, otherwise generate one new mask to use for all states in the trajectory
        if mask is None:
            my_mask = (torch.rand(self.hidden_sizes[4]) < 0.5).float().to(device) / 0.5
            # my_mask = [(torch.rand(hs) < 0.5).float().to(device) / 0.5 for hs in self.hidden_sizes] #is 0.5 too much?
        else:
            my_mask = mask

        x = traj
        x = x.view(-1,self.input_dim)

        # x = F.relu(self.fc1(x)) * my_mask[0] #mask after this, hidden_1
        # x = F.relu(self.fc2(x)) * my_mask[1] #mask after this, Hidden_2
        # x = F.relu(self.fc3(x)) * my_mask[2] #mask after this, hidden_3
        # x = F.relu(self.fc4(x)) * my_mask[3] #mask after this, hidden_4
        # x = F.relu(self.fc5(x)) * my_mask[4] #mask after this, hidden_5
        # r = self.fc6(x) #don't mask after this for now. Check with Daniel Brown

        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x)) 
        x = F.relu(self.fc3(x)) 
        x = F.relu(self.fc4(x)) 
        x = F.relu(self.fc5(x)) * my_mask
        # r = self.fc6(x) #don't mask after this for now. Check with Daniel Brown
        r = F.relu(self.fc6(x))

        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))
        return sum_rewards, sum_abs_rewards, my_mask

    def parallel_cum_return(self, traj, mask=None):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        #compute forward pass of reward network (we parallelize across frames so batch size is length of partial trajectory)
        #if mask is given then use it, otherwise generate one new mask to use for all states in the trajectory
        if mask is None:
            my_mask = (torch.rand(self.hidden_sizes[4]) < 0.5).float().to(device) / 0.5
            # my_mask = [(torch.rand(hs) < 0.5).float().to(device) / 0.5 for hs in self.hidden_sizes]
        else:
            my_mask = mask

        x = traj
        x = x.view(-1,self.input_dim)

        # x = F.relu(self.fc1(x)) * my_mask[0] #mask after this, hidden_1
        # x = F.relu(self.fc2(x)) * my_mask[1] #mask after this, Hidden_2
        # x = F.relu(self.fc3(x)) * my_mask[2] #mask after this, hidden_3
        # x = F.relu(self.fc4(x)) * my_mask[3] #mask after this, hidden_4
        # x = F.relu(self.fc5(x)) * my_mask[4] #mask after this, hidden_5
        # r = self.fc6(x) #don't mask after this for now. Check with Daniel Brown

        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x)) 
        x = F.relu(self.fc3(x)) 
        x = F.relu(self.fc4(x)) 
        x = F.relu(self.fc5(x)) * my_mask
        # r = self.fc6(x) #don't mask after this for now. Check with Daniel Brown
        r = F.relu(self.fc6(x))

        sum_rewards = r.squeeze()
        sum_abs_rewards = torch.abs(r).squeeze()
        # print('sum_rewards.size()', sum_rewards.size())
        # print('sum_rewards', sum_rewards)
        return sum_rewards, sum_abs_rewards, my_mask

    def test_cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        #compute forward pass of reward network (we parallelize across frames so batch size is length of partial trajectory)

        x = traj
        x = x.view(-1,self.input_dim)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        # r = self.fc6(x) #don't mask after this for now. Check with Daniel Brown
        r = F.relu(self.fc6(x))

        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))
        return sum_rewards, sum_abs_rewards

    def test_parallel_cum_return(self, traj, mask=None):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        #compute forward pass of reward network (we parallelize across frames so batch size is length of partial trajectory)
        #if mask is given then use it, otherwise generate one new mask to use for all states in the trajectory

        x = traj
        x = x.view(-1,self.input_dim)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        # r = self.fc6(x) #don't mask after this for now. Check with Daniel Brown
        r = F.relu(self.fc6(x))

        sum_rewards = r.squeeze()
        sum_abs_rewards = torch.abs(r).squeeze()
        # print('sum_rewards.size()', sum_rewards.size())
        # print('sum_rewards', sum_rewards)
        return sum_rewards, sum_abs_rewards

    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i, mask = self.cum_return(traj_i)
        cum_r_j, abs_r_j, _ = self.cum_return(traj_j, mask)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j
    
    def test_forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i = self.test_cum_return(traj_i)
        cum_r_j, abs_r_j = self.test_cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j