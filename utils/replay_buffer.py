import numpy as np
from gym import spaces
from typing import Union, NamedTuple


class Transition(NamedTuple):
    obs: np.ndarray
    actions: np.ndarray
    reward: np.ndarray
    dones: np.ndarray
    next_obs: np.ndarray


class ReplayBuffer(object):
    def __init__(self,
                 observation_space: Union[spaces.Box, spaces.Discrete],
                 action_space: Union[spaces.Box, spaces.Discrete],
                 buffer_size: int = 1_000_000,
                 ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space

        self.observations = self.build(self.observation_space)
        self.actions = self.build(self.action_space)
        self.next_observations = self.build(self.observation_space)
        self.rewards = np.empty(shape=(self.buffer_size, 1), dtype=np.float32)
        self.dones = np.empty(shape=(self.buffer_size, 1), dtype=np.float32)

        self.pointer = 0
        self._full = False

    @staticmethod
    def size_of_space(x: spaces.Box):
        return int(np.prod(x.shape).item())

    def build(self, space):
        if isinstance(space, spaces.Discrete):
            return np.empty(shape=(self.buffer_size, 1), dtype=np.int32)
        else:
            return np.empty(shape=(self.buffer_size, self.size_of_space(space)), dtype=np.float32)

    def append(self, obs, action, reward, done, next_obs):
        self.observations[self.pointer] = obs.copy()
        self.actions[self.pointer] = action.copy()
        self.rewards[self.pointer] = float(reward)
        self.dones[self.pointer] = float(done)
        self.next_observations[self.pointer] = next_obs.copy()

        self.pointer += 1
        self._full = (self.pointer >= self.buffer_size) or self._full
        self.pointer %= self.buffer_size

    def _len(self):
        if self._full:
            return self.buffer_size
        else:
            return self.pointer

    def __len__(self):
        return self._len()

    def _sample_index(self, batch_size: int):
        indicies = np.random.randint(low=0, high=self._len(), size=(batch_size, ))
        return indicies

    def sample(self, batch_size):
        indicies = self._sample_index(batch_size)

        obs = self.observations[indicies]
        actions = self.actions[indicies]
        rewards = self.rewards[indicies]
        dones = self.dones[indicies]
        next_obs = self.next_observations[indicies]
        return Transition(*(obs, actions, rewards, dones, next_obs))


class StackedReplayBuffer(ReplayBuffer):
    def __init__(self,
                 observation_space: Union[spaces.Box, spaces.Discrete],
                 action_space: Union[spaces.Box, spaces.Discrete],
                 buffer_size: int = 1_000_000,
                 steps: int = 20,
                 ):
        self.steps = steps
        super().__init__(
            observation_space,
            action_space,
            buffer_size
        )

        del self.observations
        del self.next_observations
        self.observations = np.empty(shape=(self.buffer_size, self.steps,
                                            int(np.prod(observation_space.shape[1:]).item())))
        self.next_observations = np.empty(shape=(self.buffer_size, self.steps,
                                                 int(np.prod(observation_space.shape[1:]).item())))

        self.pointer = 0
        self._full = False

    @staticmethod
    def size_of_space(x: spaces.Box):
        return int(np.prod(x.shape).item())