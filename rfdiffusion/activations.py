import torch
import h5py
import numpy as np
import os

class ActivationStore:
    pass

class ActivationStore2:
    def __init__(self, root_dir='./activations', compression='gzip', chunk_size=1000):
        self.root_dir = root_dir
        self.compression = compression
        self.chunk_size = chunk_size
        os.makedirs(root_dir, exist_ok=True)
        self.current_file = None
        self.call_count = 0
        
    def store_activations(self, cache, iteration):
        """Store the current activation cache to disk"""
        if self.current_file is None or self.call_count % self.chunk_size == 0:
            # Create a new file for every chunk_size iterations
            if self.current_file is not None:
                self.current_file.close()
            file_path = os.path.join(self.root_dir, f'activations_{self.call_count//self.chunk_size}.h5')
            self.current_file = h5py.File(file_path, 'w')
            
        # Store the activations for this iteration
        grp = self.current_file.create_group(f'iter_{iteration}')
        for name, tensor in cache.items():
            # Convert to numpy and store with compression
            if tensor.is_sparse:
                # Handle sparse tensors specially
                indices = tensor._indices().numpy()
                values = tensor._values().numpy()
                shape = tensor.shape
                grp.create_dataset(f'{name}_indices', data=indices, compression=self.compression)
                grp.create_dataset(f'{name}_values', data=values, compression=self.compression)
                grp.create_dataset(f'{name}_shape', data=np.array(shape))
            else:
                # Convert dense tensor to numpy and store
                grp.create_dataset(name, data=tensor.cpu().numpy(), compression=self.compression)
        
        self.call_count += 1
        
    def close(self):
        if self.current_file is not None:
            self.current_file.close()
