import random
from collections import deque


class ReplayBuffer:
    """A replay buffer as commonly used for off-policy Q-Learning methods.

    This buffer stores transitions (state, action, reward, next_state, terminated) and 
    allows sampling random batches for training. It has a fixed capacity and 
    automatically overwrites the oldest entries when full.
    """

    def __init__(self, capacity):
        """Initialize the replay buffer with a fixed capacity.

        Parameters
        ----------
        capacity : int
            The maximum number of transitions that can be stored in the buffer.
            When the buffer is full, adding new transitions will overwrite the oldest ones.
        """
        self.buffer = deque(maxlen=capacity)

    def put(self, s, a, r, s_, terminated):
        """Put a tuple of (obs, action, rewards, next_obs, terminated) into the replay buffer.
        The max length specified by capacity should never be exceeded.
        The oldest elements inside the replay buffer should be overwritten first.

        Parameters
        ----------
        s : numpy.ndarray
            The current state/observation.
        a : int
            The action taken.
        r : float
            The reward received.
        s_ : numpy.ndarray
            The next state/observation.
        terminated : bool
            Whether the episode terminated.
        """

        self.buffer.append((s, a, r, s_, terminated))

    def get(self, batch_size):
        """Gives batch_size samples from the replay buffer.

        Parameters
        ----------
        batch_size : int
            Number of samples to return from the buffer.

        Returns
        -------
        list
            A list of (state, action, reward, next_state, terminated) tuples.
        """

        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Returns the number of tuples inside the replay buffer.

        Returns
        -------
        int
            The current number of tuples in the buffer.
        """

        return len(self.buffer)
