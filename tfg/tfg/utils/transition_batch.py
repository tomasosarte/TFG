import threading
import torch as th
import numpy as np

class TransitionBatch:
    """ Simple implementation of a batchof transitionsm (or another dictionary-based tensor structure).
        Read and write operations are thread-safe, but the iterator is not (you cannot interate
        over the same TransitionBatch in two threads at the same time). """
    def __init__(self, max_size: int, transition_format: dict, batch_size: int=32) -> None:
        """
        Initializes the TransitionBatch object.

        Args:
            max_size (int): The maximum size of the batch.
            transition_format (dict): A dictionary containing the format of the transitions.
            batch_size (int): The batch size.
        
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
            self.dict[key] = th.zeros([max_size, *spec[0]], dtype=spec[1])
            
    def _clone_empty_batch(self, max_size: int=None, batch_size: int=None) -> 'TransitionBatch':
        """ 
        Clones this TransitionBatch without cloning the data. 
        
        Args:
            max_size (int): The maximum size of the batch.
            batch_size (int): The batch size.
            
        Returns:
            TransitionBatch: The cloned TransitionBatch.
        """
        max_size = self.max_size if max_size is None else max_size
        batch_size = self.batch_size if batch_size is None else batch_size
        return TransitionBatch(max_size=max_size, transition_format={}, batch_size=batch_size)
    
    def to(self, device: th.device) -> 'TransitionBatch':
        """ 
        Move all tensors in the dictionary to the specified device. 
        
        Args:
            device: The device to move the tensors to.
        
        Returns:
            TransitionBatch: The TransitionBatch with the tensors moved to the specified device.
        """
        self.lock.acquire()
        try:
            for key in self.dict.keys():
                self.dict[key] = self.dict[key].to(device)
        finally:
            self.lock.release()
        return self
    
    def __getitem__(self, key) -> 'TransitionBatch':
        """ 
        Access the TransitionBatch with the [] operator. Use as key either 
        - the string name of a variable to get the full tensor of that variable,
        - a slice to get a time-slice over all variables in the batch,
        - a LongTensor that selects a subset of indices for all variables in the batch. 
        
        Args:
            key: The key to access the TransitionBatch.
            
        Returns:
            The entry of the transition called "key", a slice"
        """
        # Return the entry of the transition called "key"
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
        # Collect and return a set of transitions specified by the LongTensor "key" 
        if isinstance(key, th.Tensor):
            self.lock.acquire()
            try:
                batch = self._clone_empty_batch(max_size=key.shape[0])
                batch.size = key.shape[0]
                for k, v in self.dict.items():
                    key = key.view(batch.size, *[1 for _ in range(len(v.shape[1:]))])
                    batch.dict[k] = v.gather(dim=0, index=key.expand(batch.size, *v.shape[1:]))
            finally: self.lock.release()
            return batch
        return None
    
    def get_first(self) -> 'TransitionBatch':
        """ 
        Returns a batch of the oldest entries of all variables. 
        
        Args:
            None

        Returns:
            TransitionBatch: A batch of the oldest entries of all variables.
        """
        batch = self._clone_empty_batch(max_size=1)
        self.lock.acquire()
        try:
            batch.size = 1
            for k, v in self.dict.items():
                batch.dict[k] = v[self.first].unsqueeze(dim=0)
        finally: self.lock.release()
        return batch    
    
    def get_last(self) -> 'TransitionBatch':
        """ 
        Returns a batch of the newest entries of all variables. 
        
        Args:
            None
        
        Returns:
            TransitionBatch: A batch of the newest entries of all variables.
        """
        batch = self._clone_empty_batch(max_size=1)
        self.lock.acquire()
        try:
            batch.size = 1
            for k, v in self.dict.items():
                batch.dict[k] = v[(self.first + self.size - 1) % self.size].unsqueeze(dim=0)
        finally: self.lock.release()
        return batch
    
    def add(self, trans: dict) -> 'TransitionBatch':
        """ 
        Adding transition dictionaries, which can contain Tensors of arbitrary length. 
        
        Args:
            trans (dict): The transition dictionary to add.
        
        Returns:
            TransitionBatch: The TransitionBatch with the added transition dictionary.
        """
        if isinstance(trans, TransitionBatch):
            trans = trans.dict
        # Add all data in the dict
        self.lock.acquire()
        try:
            n = 0
            idx = None
            for k, v in trans.items():
                if idx is None:
                    n = v.shape[0]
                    idx = th.tensor([(self.first + self.size + i) % self.max_size for i in range(n)], dtype=th.long)
                else:
                    assert n == v.shape[0], 'all tensors in a transition need to have the same batch_size'
                idx = idx.view(idx.shape[0], *[1 for _ in range(len(v.shape) - 1)])
                self.dict[k].scatter_(dim=0, index=idx.expand_as(v), src=v)
            # Increase the size (and handle overflow)
            self.size += n
            if self.size > self.max_size:
                self.first = (self.first + n) % self.max_size
                self.size = self.max_size
        finally: self.lock.release()
        return self
            
    def trim(self) -> 'TransitionBatch':
        """ 
        Reduces the length of the max_size to its actual size (in-place). Returns self. 
        
        Args:
            None
        
        Returns:
            TransitionBatch: The TransitionBatch with the reduced length of the max_size.
        """
        self.lock.acquire()
        try:
            for k, v in self.dict.items():
                self.dict[k] = v[:self.size]
            self.max_size = self.size
        finally: self.lock.release()
        return self
    
    def replace(self, batch: 'TransitionBatch', index: int) -> None:
        """ 
        Replaces parts of this batch with another batch (which must be smaller). 
        
        Args:
            batch: The batch to replace.
            index (int): The index to replace.
        
        Returns:
            None
        """
        self.lock.acquire()
        try:
            #assert batch.max_size <= self.max_size - index, "Replacement is larger then target area in batch."
            assert batch.size <= self.max_size - index, "Replacement is larger then target area in batch."
            for k, v in batch.dict.items():
                if batch.size < batch.max_size:
                    v = v[:batch.size]
                self.dict[k][index:(index + batch.max_size)] = v    
        finally: self.lock.release()
    
    def sample(self) -> 'TransitionBatch':
        """ 
        Samples a random mini-batch from the batch. 
        
        Args:
            None
        
        Returns:
            TransitionBatch: A random mini-batch from the batch.
        """
        return self[th.randint(high=self.size, size=(self.batch_size,1))]
            
    def __len__(self) -> int:
        """ 
        Returns the length of the batch. 
        
        Args:
            None
        
        Returns:
            int: The length of the batch.
        """
        return self.size
    
    def __iter__(self) -> 'TransitionBatch':
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
            list: A list of contiguous tensors.
        """
        if len(self.indices) == 0: raise StopIteration
        size = min(self.batch_size, len(self.indices))
        batch = self[th.LongTensor(self.indices[-size:])]
        self.indices = self.indices[:-size]
        return batch