import os
import gym
import h5py
import urllib.request
import numpy as np
import deepdish as dd

def set_dataset_path(path):
    global DATASET_PATH
    DATASET_PATH = path
    os.makedirs(path, exist_ok=True)

set_dataset_path(os.environ.get('D4RL_DATASET_DIR', os.path.expanduser('~/.d4rl/datasets')))

def get_keys(h5file):
    keys = []
    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)
    h5file.visititems(visitor)
    return keys


def filepath_from_url(dataset_url):
    _, dataset_name = os.path.split(dataset_url)
    dataset_filepath = os.path.join(DATASET_PATH, dataset_name)
    return dataset_filepath


def download_dataset_from_url(dataset_url):
    dataset_filepath = filepath_from_url(dataset_url)
    if not os.path.exists(dataset_filepath):
        print('Downloading dataset:', dataset_url, 'to', dataset_filepath)
        urllib.request.urlretrieve(dataset_url, dataset_filepath)
    if not os.path.exists(dataset_filepath):
        raise IOError("Failed to download dataset from %s" % dataset_url)
    return dataset_filepath



class OfflineEnv(gym.Env):
    """
    Base class for offline RL envs.
    Args:
        dataset_url: URL pointing to the dataset.
        ref_max_score: Maximum score (for score normalization)
        ref_min_score: Minimum score (for score normalization)
    """
    def __init__(self, dataset_url=None, ref_max_score=None, ref_min_score=None, **kwargs):
        super(OfflineEnv, self).__init__(**kwargs)
        self.dataset_url = self._dataset_url = dataset_url
        # print('self.dataset_url', self.dataset_url)
        env_name = os.path.basename(self.dataset_url)
        self.env_prefix = env_name.split('-')[0]
        # print('env_prefix', env_prefix)
        # import pdb; pdb.set_trace()
        self.ref_max_score = ref_max_score
        self.ref_min_score = ref_min_score
        self.set_mask_flag = False
        self.reward_model_path = None
        # self.random_data = False
        # self.set_no_terminal = False
    
    def set_mask(self, mask_ratio=0.0, mask_with_zero=True):
        self.mask_ratio = mask_ratio
        self.mask_with_zero = mask_with_zero
        self.set_mask_flag = True

    def set_reward_path(self, reward_model_path):
        self.reward_model_path = reward_model_path

    def set_terminal(self, no_terminal):
        self.no_terminal = no_terminal
        self.set_no_terminal = True
    
    def set_random_data(self):
        self.random_data = True

    def get_normalized_score(self, score):
        if (self.ref_max_score is None) or (self.ref_min_score is None):
            raise ValueError("Reference score not provided for env")
        return (score - self.ref_min_score) / (self.ref_max_score - self.ref_min_score)

    @property
    def dataset_filepath(self):
        return filepath_from_url(self.dataset_url)

    def get_dataset(self, mask_ratio=0.0, mask_with_zero=True, mask_seed=0, reward_model_path=None, h5path=None):
    # def get_dataset(self, h5path=None):

        if self.set_mask_flag:
            mask_ratio = self.mask_ratio
            mask_with_zero = self.mask_with_zero

        if h5path is None:
            if self._dataset_url is None:
                raise ValueError("Offline env not configured with a dataset URL.")
            h5path = download_dataset_from_url(self.dataset_url)

        dataset_file = h5py.File(h5path, 'r')
        data_dict = {}
        for k in get_keys(dataset_file):
            # print('k', k)
            try:
                # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e: # try loading as a scalar
                data_dict[k] = dataset_file[k][()]
        dataset_file.close()

        # Run a few quick sanity checks
        for key in ['observations', 'actions', 'rewards', 'terminals']:
            assert key in data_dict, 'Dataset is missing key %s' % key
        N_samples = data_dict['observations'].shape[0]
        if self.observation_space.shape is not None:
            assert data_dict['observations'].shape[1:] == self.observation_space.shape, \
                    'Observation shape does not match env: %s vs %s' % (str(data_dict['observations'].shape[1:]), str(self.observation_space.shape))
        assert data_dict['actions'].shape[1:] == self.action_space.shape, \
                    'Action shape does not match env: %s vs %s' % (str(data_dict['actions'].shape[1:]), str(self.action_space.shape))
        if data_dict['rewards'].shape == (N_samples, 1):
            data_dict['rewards'] = data_dict['rewards'][:,0]
        assert data_dict['rewards'].shape == (N_samples,), 'Reward has wrong shape: %s' % (str(data_dict['rewards'].shape))
        if data_dict['terminals'].shape == (N_samples, 1):
            data_dict['terminals'] = data_dict['terminals'][:,0]
        assert data_dict['terminals'].shape == (N_samples,), 'Terminals has wrong shape: %s' % (str(data_dict['rewards'].shape))

        num_samples = data_dict['rewards'].shape[0]
        import math
        num_mask = math.floor(mask_ratio*num_samples)
        zero_mask = np.ones(num_samples)
        zero_mask[:num_mask] = 0

        #seed for different mask ordering
        np.random.seed(mask_seed)
        np.random.shuffle(zero_mask)
        mean_reward = np.mean(data_dict['rewards'])

        if mask_with_zero:
            mean_reward = 0.0

        data_dict['rewards'][zero_mask == 0] = mean_reward

        if reward_model_path:
            print('in side of if not reward_model_path loop')
            with open(f'{reward_model_path}', 'rb') as f:
                reward_arr = np.load(f)
            data_dict['rewards'] = reward_arr

        assert data_dict['rewards'].shape[0] == data_dict['observations'].shape[0]
        return data_dict


    def get_dataset_chunk(self, chunk_id, h5path=None):
        """
        Returns a slice of the full dataset.
        Args:
            chunk_id (int): An integer representing which slice of the dataset to return.
        Returns:
            A dictionary containing observtions, actions, rewards, and terminals.
        """
        if h5path is None:
            if self._dataset_url is None:
                raise ValueError("Offline env not configured with a dataset URL.")
            h5path = download_dataset_from_url(self.dataset_url)

        dataset_file = h5py.File(h5path, 'r')

        if 'virtual' not in dataset_file.keys():
            raise ValueError('Dataset is not a chunked dataset')
        available_chunks = [int(_chunk) for _chunk in list(dataset_file['virtual'].keys())]
        if chunk_id not in available_chunks:
            raise ValueError('Chunk id not found: %d. Available chunks: %s' % (chunk_id, str(available_chunks)))

        load_keys = ['observations', 'actions', 'rewards', 'terminals']
        data_dict = {k: dataset_file['virtual/%d/%s' % (chunk_id, k)][:] for k in load_keys}
        dataset_file.close()
        return data_dict


class OfflineEnvWrapper(gym.Wrapper, OfflineEnv):
    """
    Wrapper class for offline RL envs.
    """
    def __init__(self, env, **kwargs):
        gym.Wrapper.__init__(self, env)
        OfflineEnv.__init__(self, **kwargs)

    def reset(self):
        return self.env.reset()