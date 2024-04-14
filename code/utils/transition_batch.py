import threading
import torch as th
import numpy as np
class TransitionBatch:
    """
    A class to represent a batch of transitions.
    """

    def __init__(self, max_size: int, transition_format: dict, batch_size=32):
        """
        Constructor for the TransitionBatch class.
        Args:
            max_size (int): The maximum size of the batch.
            transition_format (dict): A dictionary containing the format of the transitions.
            batch_size (int): The batch size of the transitions.
        Returns:
            None
        """
        self.lock = threading.Lock()
        self.indices = []
        self.size = 0
        self.first = 0
        self.max_size = max_size
        self.batch_size = batch_size
        self.dict = {}
        for key, spec in transition_format.items():
            # if spec[1] is dict:
            #     # Fill the dictionary with a list of empty dictionaries
            #     self.dict[key] = [{} for _ in range(max_size)]
            # else:
            self.dict[key] = th.zeros([max_size, *spec[0]], dtype=spec[1])

    def _clone_empty_batch(self, max_size: int = None, batch_size: int = None):
        """ 
        Clones this TransitionBatch without cloning the data. 
        
        Args:
            max_size (int): The maximum size of the new TransitionBatch.
            batch_size (int): The batch size of the new TransitionBatch.
        
        Returns:
            TransitionBatch: A new TransitionBatch with the same format but no data.
        """
        max_size = self.max_size if max_size is None else max_size
        batch_size = self.batch_size if batch_size is None else batch_size
        return TransitionBatch(max_size=max_size, transition_format={}, batch_size=batch_size)

    def __getitem__(self, key):
        """
        Access the TransitionBatch with the [] operator. use as key either
        - the string name of a variable to get full tensor of that variable
        - slice to get a time-slice over all variables in the batch
        - a LongTensor to get a set of transitions specified by the indices in the LongTensor
        Args:
            key (str, slice, LongTensor): The key to access the TransitionBatch.
        
        Returns:
            th.Tensor: The tensor corresponding to the key.
        """
        # Return the entry of the transition called 'key
        if isinstance(key, str):
            return self.dict[key]
        # Return a slice of the batch
        if isinstance(key, slice):
            key = slice(0 if key.start is None else key.start, self.size if key.stop is None else key.stop, 
                        1 if key.step is None else key.step)
            self.lock.acquire()
            try:
                batch = self._clone_empty_batch()
                batch.size = (key.stop - key.start) // key.step
                for k, v in self.dict.items():
                    batch.dict[k] = v[key]
            finally: self.lock.release()
            return batch
        # Collect and return a set of transitions specified by a LongTensor 'key'
        if isinstance(key, th.LongTensor):
            self.lock.acquire()
            try:
                batch = self._clone_empty_batch(max_size=key.size(0))
                batch.size = key.shape[0]
                for k, v in self.dict.items():
                    key = key.view(batch.size, *[1 for _ in range(len(v.shape[1:]))])
            finally: self.lock.release()
            return batch
        return None

    def __len__(self): 
        """ 
        Returns the length of the batch. 
        
        Args:
            None
        
        Returns:
            int: The length of the batch.
        """
        return self.size

    def __iter__(self):  
        """ 
        Initializes an iterator over the batch. 
        
        Args:
            None
        
        Returns:
            TransitionBatch: The iterator over the batch.
        """
        self.indices = list(range(self.size))
        np.random.shuffle(self.indices)
        return self

    def __next__(self):  
        """ 
        Iterates through batch, returns list of contiguous tensors. 
        
        Args:
            None
        
        Returns:
            TransitionBatch: A list of contiguous tensors.
        """
        if len(self.indices) == 0: raise StopIteration
        size = min(self.batch_size, len(self.indices))
        batch = self[th.LongTensor(self.indices[-size:])]
        self.indices = self.indices[:-size]
        return batch

    def trim(self):
        """ 
        Reduces the length of the max_size to its actual size (in-place). Returns self. 
        
        Args:
            None
        
        Returns:
            TransitionBatch: The TransitionBatch with the length of the max_size reduced to its actual size.
        """
        self.lock.acquire()
        try:
            for k, v in self.dict.items():
                self.dict[k] = v[:self.size]
            self.max_size = self.size
        finally: self.lock.release()
        return self  

    def add(self, transition: dict):
        """ 
        Adding transition dictionaries, which can contain Tensors of arbitrary length. 
        
        Args:
            transition (dict): The transition dictionary to add.
        
        Returns:
            TransitionBatch: The TransitionBatch with the added transition dictionary.
        """
        # Add all data in the dict
        self.lock.acquire()
        try:
            idx = None
            n = 1
            for k, v in transition.items():
                if idx is None: idx = th.LongTensor([(self.first + self.size) % self.max_size])
                self.dict[k][idx] = v
            # Increase the size (and handle overflow)
            self.size += n
            if self.size > self.max_size:
                self.first = (self.first + n) % self.max_size
                self.size = self.max_size
        finally: self.lock.release()
        return self