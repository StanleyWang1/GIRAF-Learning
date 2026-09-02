import numpy as np
from typing import Union, Dict, Optional
import os
import math
import numbers
import zarr
import numcodecs
import numpy as np
from functools import cached_property

try:
    import pyrealsense2 as rs  # type: ignore
except Exception:
    rs = None

try:
    from hydra import compose, initialize_config_dir  # type: ignore
except Exception:
    compose = None
    initialize_config_dir = None

def load_hydra_config_with_defaults(config_path):
    """
    Load config using Hydra's compose API to properly resolve defaults.
    """
    if compose is None or initialize_config_dir is None:
        raise ImportError("Hydra is required for load_hydra_config_with_defaults but is not installed")

    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)
    config_name = os.path.basename(config_path)
    if config_name.endswith('.yaml'):  # Strip extension
        config_name = config_name[:-5]
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
    return cfg

class NumpyAccumulator:
    def __init__(self, initial_capacity=300, shape_suffix=(), dtype=np.float32):
        """
        initial_capacity: starting number of rows
        shape_suffix: trailing dimensions, e.g. (3,) for RGB, (4,4) for matrices
        dtype: dtype of the array
        """
        self.capacity = initial_capacity
        self._initial_capacity = initial_capacity  # Store for reset purposes
        self.size = 0   # number of actual elements stored
        
        self.shape_suffix = shape_suffix
        self.dtype = dtype

        self.buffer = np.zeros((self.capacity,) + shape_suffix, dtype=dtype)

    def __len__(self):
        return self.size

    def _ensure_capacity(self, needed_size):
        if needed_size <= self.capacity:
            return

        # Double until large enough
        new_capacity = self.capacity
        while new_capacity < needed_size:
            new_capacity *= 2

        new_buffer = np.zeros((new_capacity,) + self.shape_suffix, dtype=self.dtype)
        new_buffer[:self.size] = self.buffer[:self.size]
        
        self.buffer = new_buffer
        self.capacity = new_capacity

    def append(self, value):
        """Append a single item."""
        self._ensure_capacity(self.size + 1)
        self.buffer[self.size] = value
        self.size += 1

    def extend(self, values):
        """
        Append multiple values at once.
        values: np.ndarray with shape (N, *shape_suffix)
        """
        values = np.asarray(values)
        n = len(values)
        self._ensure_capacity(self.size + n)

        self.buffer[self.size:self.size+n] = values
        self.size += n

    @property
    def data(self):
        """Return a view of the valid portion."""
        return self.buffer[:self.size]
    
    def reset(self):
        self.size = 0
        # Reset capacity to initial size and create new buffer
        self.capacity = self._initial_capacity
        self.buffer = np.zeros((self.capacity,) + self.shape_suffix, dtype=self.dtype)



def check_chunks_compatible(chunks: tuple, shape: tuple):
    assert len(shape) == len(chunks)
    for c in chunks:
        assert isinstance(c, numbers.Integral)
        assert c > 0

def rechunk_recompress_array(group, name, 
        chunks=None, chunk_length=None,
        compressor=None, tmp_key='_temp'):
    old_arr = group[name]
    if chunks is None:
        if chunk_length is not None:
            chunks = (chunk_length,) + old_arr.chunks[1:]
        else:
            chunks = old_arr.chunks
    check_chunks_compatible(chunks, old_arr.shape)
    
    if compressor is None:
        compressor = old_arr.compressor
    
    if (chunks == old_arr.chunks) and (compressor == old_arr.compressor):
        # no change
        return old_arr

    # rechunk recompress
    group.move(name, tmp_key)
    old_arr = group[tmp_key]
    n_copied, n_skipped, n_bytes_copied = zarr.copy(
        source=old_arr,
        dest=group,
        name=name,
        chunks=chunks,
        compressor=compressor,
    )
    del group[tmp_key]
    arr = group[name]
    return arr

def get_optimal_chunks(shape, dtype, 
        target_chunk_bytes=2e6, 
        max_chunk_length=None):
    """
    Common shapes
    T,D
    T,N,D
    T,H,W,C
    T,N,H,W,C
    """
    itemsize = np.dtype(dtype).itemsize
    # reversed
    rshape = list(shape[::-1])
    if max_chunk_length is not None:
        rshape[-1] = int(max_chunk_length)
    split_idx = len(shape)-1
    for i in range(len(shape)-1):
        this_chunk_bytes = itemsize * np.prod(rshape[:i])
        next_chunk_bytes = itemsize * np.prod(rshape[:i+1])
        if this_chunk_bytes <= target_chunk_bytes \
            and next_chunk_bytes > target_chunk_bytes:
            split_idx = i

    rchunks = rshape[:split_idx]
    item_chunk_bytes = itemsize * np.prod(rshape[:split_idx])
    this_max_chunk_length = rshape[split_idx]
    next_chunk_length = min(this_max_chunk_length, math.ceil(
            target_chunk_bytes / item_chunk_bytes))
    rchunks.append(next_chunk_length)
    len_diff = len(shape) - len(rchunks)
    rchunks.extend([1] * len_diff)
    chunks = tuple(rchunks[::-1])
    # print(np.prod(chunks) * itemsize / target_chunk_bytes)
    return chunks


class ReplayBuffer:
    """
    Zarr-based temporal datastructure.
    Assumes first dimension to be time. Only chunk in time dimension.
    """
    def __init__(self, 
            root: Union[zarr.Group, 
            Dict[str,dict]]):
        """
        Dummy constructor. Use copy_from* and create_from* class methods instead.
        """
        assert('data' in root)
        assert('meta' in root)
        assert('episode_ends' in root['meta'])
        self.root = root
        
        # Only stabilize if not read-only
        if not self._is_read_only():
            self.stabilize()
        
        for key, value in root['data'].items():
            assert(value.shape[0] == root['meta']['episode_ends'][-1])
        # print(f"Contains {len(self.episode_ends)-1} (0-indexed) episodes!")
    
    def _is_read_only(self) -> bool:
        """Check if the underlying store is read-only."""
        if isinstance(self.root, zarr.Group):
            store = self.root.store
            # Check for read_only attribute on store
            if hasattr(store, 'read_only'):
                return store.read_only
            # For DirectoryStore, try to infer from mode if available
            if hasattr(store, 'mode'):
                return store.mode == 'r'
        return False
        
    
    # ============= create constructors ===============
    @classmethod
    def create_empty_zarr(cls, storage=None, root=None):
        if root is None:
            if storage is None:
                storage = zarr.MemoryStore()
            root = zarr.group(store=storage)
        data = root.require_group('data', overwrite=False)
        meta = root.require_group('meta', overwrite=False)
        if 'episode_ends' not in meta:
            episode_ends = meta.zeros('episode_ends', shape=(0,), dtype=np.int64,
                compressor=None, overwrite=False)
        return cls(root=root)
    
    @classmethod
    def create_empty_numpy(cls):
        root = {
            'data': dict(),
            'meta': {
                'episode_ends': np.zeros((0,), dtype=np.int64)
            }
        }
        return cls(root=root)
    
    @classmethod
    def create_from_group(cls, group, **kwargs):
        if 'data' not in group:
            # create from stratch
            buffer = cls.create_empty_zarr(root=group, **kwargs)
        else:
            # already exist
            buffer = cls(root=group, **kwargs)
        return buffer

    @classmethod
    def create_from_path(cls, zarr_path, mode='r', **kwargs):
        """
        Open a on-disk zarr directly (for dataset larger than memory).
        Slower.
        """
        group = zarr.open(os.path.expanduser(zarr_path), mode)
        return cls.create_from_group(group, **kwargs)
    
    # ============= copy constructors ===============
    @classmethod
    def copy_from_store(cls, src_store, store=None, keys=None, 
            chunks: Dict[str,tuple]=dict(), 
            compressors: Union[dict, str, numcodecs.abc.Codec]=dict(), 
            if_exists='replace',
            **kwargs):
        """
        Load to memory.
        """
        src_root = zarr.group(src_store)
        root = None
        if store is None:
            # numpy backend
            meta = dict()
            for key, value in src_root['meta'].items():
                if len(value.shape) == 0:
                    meta[key] = np.array(value)
                else:
                    meta[key] = value[:]

            if keys is None:
                keys = src_root['data'].keys()
            data = dict()
            for key in keys:
                arr = src_root['data'][key]
                data[key] = arr[:]

            root = {
                'meta': meta,
                'data': data
            }
        else:
            root = zarr.group(store=store)
            # copy without recompression
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(source=src_store, dest=store,
                source_path='/meta', dest_path='/meta', if_exists=if_exists)
            data_group = root.create_group('data', overwrite=True)
            if keys is None:
                keys = src_root['data'].keys()
            for key in keys:
                value = src_root['data'][key]
                cks = cls._resolve_array_chunks(
                    chunks=chunks, key=key, array=value)
                cpr = cls._resolve_array_compressor(
                    compressors=compressors, key=key, array=value)
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = '/data/' + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=src_store, dest=store,
                        source_path=this_path, dest_path=this_path,
                        if_exists=if_exists
                    )
                else:
                    # copy with recompression
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value, dest=data_group, name=key,
                        chunks=cks, compressor=cpr, if_exists=if_exists
                    )
        buffer = cls(root=root)
        return buffer
    
    @classmethod
    def copy_from_path(cls, zarr_path, backend=None, store=None, keys=None, 
            chunks: Dict[str,tuple]=dict(), 
            compressors: Union[dict, str, numcodecs.abc.Codec]=dict(), 
            if_exists='replace',
            **kwargs):
        """
        Copy a on-disk zarr to in-memory compressed.
        Recommended
        """
        if backend == 'numpy':
            print('backend argument is deprecated!')
            store = None
        group = zarr.open(os.path.expanduser(zarr_path), 'r')
        return cls.copy_from_store(src_store=group.store, store=store, 
            keys=keys, chunks=chunks, compressors=compressors, 
            if_exists=if_exists, **kwargs)

    # ============= save methods ===============
    def save_to_store(self, store, 
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict(),
            if_exists='replace', 
            **kwargs):
        
        root = zarr.group(store)
        if self.backend == 'zarr':
            # recompression free copy
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                source=self.root.store, dest=store,
                source_path='/meta', dest_path='/meta', if_exists=if_exists)
        else:
            meta_group = root.create_group('meta', overwrite=True)
            # save meta, no chunking
            for key, value in self.root['meta'].items():
                _ = meta_group.array(
                    name=key,
                    data=value, 
                    shape=value.shape, 
                    chunks=value.shape)
        
        # save data, chunk
        data_group = root.create_group('data', overwrite=True)
        for key, value in self.root['data'].items():
            cks = self._resolve_array_chunks(
                chunks=chunks, key=key, array=value)
            cpr = self._resolve_array_compressor(
                compressors=compressors, key=key, array=value)
            if isinstance(value, zarr.Array):
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = '/data/' + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=self.root.store, dest=store,
                        source_path=this_path, dest_path=this_path, if_exists=if_exists)
                else:
                    # copy with recompression
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value, dest=data_group, name=key,
                        chunks=cks, compressor=cpr, if_exists=if_exists
                    )
            else:
                # numpy
                _ = data_group.array(
                    name=key,
                    data=value,
                    chunks=cks,
                    compressor=cpr
                )
        return store

    def save_to_path(self, zarr_path,             
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict(), 
            if_exists='replace', 
            **kwargs):
        store = zarr.DirectoryStore(os.path.expanduser(zarr_path))
        return self.save_to_store(store, chunks=chunks, 
            compressors=compressors, if_exists=if_exists, **kwargs)

    @staticmethod
    def resolve_compressor(compressor='default'):
        if compressor == 'default':
            compressor = numcodecs.Blosc(cname='lz4', clevel=5, 
                shuffle=numcodecs.Blosc.NOSHUFFLE)
        elif compressor == 'disk':
            compressor = numcodecs.Blosc('zstd', clevel=5, 
                shuffle=numcodecs.Blosc.BITSHUFFLE)
        return compressor

    @classmethod
    def _resolve_array_compressor(cls, 
            compressors: Union[dict, str, numcodecs.abc.Codec], key, array):
        # allows compressor to be explicitly set to None
        cpr = 'nil'
        if isinstance(compressors, dict):
            if key in compressors:
                cpr = cls.resolve_compressor(compressors[key])
            elif isinstance(array, zarr.Array):
                cpr = array.compressor
        else:
            cpr = cls.resolve_compressor(compressors)
        # backup default
        if cpr == 'nil':
            cpr = cls.resolve_compressor('default')
        return cpr
    
    @classmethod
    def _resolve_array_chunks(cls,
            chunks: Union[dict, tuple], key, array):
        cks = None
        if isinstance(chunks, dict):
            if key in chunks:
                cks = chunks[key]
            elif isinstance(array, zarr.Array):
                cks = array.chunks
        elif isinstance(chunks, tuple):
            cks = chunks
        else:
            raise TypeError(f"Unsupported chunks type {type(chunks)}")
        # backup default
        if cks is None:
            cks = get_optimal_chunks(shape=array.shape, dtype=array.dtype)
        # check
        check_chunks_compatible(chunks=cks, shape=array.shape)
        return cks
    
    # ============= properties =================
    @cached_property
    def data(self):
        return self.root['data']
    
    @cached_property
    def meta(self):
        return self.root['meta']

    def update_meta(self, data):
        # sanitize data
        np_data = dict()
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                np_data[key] = value
            else:
                arr = np.array(value)
                if arr.dtype == object:
                    raise TypeError(f"Invalid value type {type(value)}")
                np_data[key] = arr

        meta_group = self.meta
        if self.backend == 'zarr':
            for key, value in np_data.items():
                _ = meta_group.array(
                    name=key,
                    data=value, 
                    shape=value.shape, 
                    chunks=value.shape,
                    overwrite=True)
        else:
            meta_group.update(np_data)
        
        return meta_group
    
    @property
    def episode_ends(self):
        return self.meta['episode_ends']
    
    def get_episode_idxs(self):
        import numba
        numba.jit(nopython=True)
        def _get_episode_idxs(episode_ends):
            result = np.zeros((episode_ends[-1],), dtype=np.int64)
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i-1]
                end = episode_ends[i]
                for idx in range(start, end):
                    result[idx] = i
            return result
        return _get_episode_idxs(self.episode_ends)
        
    
    @property
    def backend(self):
        backend = 'numpy'
        if isinstance(self.root, zarr.Group):
            backend = 'zarr'
        return backend
    
    # =========== dict-like API ==============
    def __repr__(self) -> str:
        if self.backend == 'zarr':
            return str(self.root.tree())
        else:
            return super().__repr__()

    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def items(self):
        return self.data.items()
    
    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    # =========== our API ==============
    @property
    def n_steps(self):
        if len(self.episode_ends) == 0:
            return 0
        return self.episode_ends[-1]
    
    @property
    def n_episodes(self):
        return len(self.episode_ends)

    @property
    def chunk_size(self):
        if self.backend == 'zarr':
            return next(iter(self.data.arrays()))[-1].chunks[0]
        return None

    @property
    def episode_lengths(self):
        ends = self.episode_ends[:]
        ends = np.insert(ends, 0, 0)
        lengths = np.diff(ends)
        return lengths
    
    def add_episode_separate(self, 
            data: Dict[str, np.ndarray], 
            chunks: Optional[Dict[str, tuple]] = dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict] = dict(),
            finalize_episode: bool = True):
            """
            Adds an episode to the buffer.
            
            Args:
                data: Dictionary of arrays to add.
                chunks: Chunking configuration for Zarr.
                compressors: Compressor configuration for Zarr.
                finalize_episode: If True, updates 'episode_ends' to mark the episode as complete. 
                                If False, writes data to arrays but keeps the episode 'open' 
                                (does not increment n_steps). Use False for the first N-1 processes 
                                and True for the final process.
            """
            assert(len(data) > 0)
            is_zarr = (self.backend == 'zarr')

            # Get current global length (based on the last finalized episode)
            # If finalize_episode was False in a previous call, n_steps has NOT moved yet,
            # so we correctly overwrite/append to the same index range.
            curr_len = self.n_steps
            
            episode_length = None
            for key, value in data.items():
                assert(len(value.shape) >= 1)
                if episode_length is None:
                    episode_length = len(value)
                else:
                    assert(episode_length == len(value))
                    
            # Calculate the target length for this specific write
            new_len = curr_len + episode_length

            for key, value in data.items():
                new_shape = (new_len,) + value.shape[1:]
                
                # 1. Create array if it doesn't exist
                if key not in self.data:
                    if is_zarr:
                        cks = self._resolve_array_chunks(
                            chunks=chunks, key=key, array=value)
                        cpr = self._resolve_array_compressor(
                            compressors=compressors, key=key, array=value)
                        arr = self.data.zeros(name=key, 
                            shape=new_shape, 
                            chunks=cks,
                            dtype=value.dtype,
                            compressor=cpr)
                    else:
                        # copy data to prevent modify
                        arr = np.zeros(shape=new_shape, dtype=value.dtype)
                        self.data[key] = arr
                else:
                    # 2. Resize existing array
                    arr = self.data[key]
                    assert(value.shape[1:] == arr.shape[1:])
                    
                    # Only resize if the array isn't already large enough.
                    # This handles cases where Process A (Action) resized it to new_len,
                    # and now Process B (Image) sees it's already big enough.
                    if arr.shape[0] < new_len:
                        if is_zarr:
                            arr.resize(new_shape)
                        else:
                            arr.resize(new_shape, refcheck=False)
                            
                # 3. Copy data to the specific slice
                # We write to the range [curr_len : new_len]
                arr[curr_len:new_len] = value
            
            # Only update the global episode tracker if this is the generic "finalizing" call
            if finalize_episode:
                # append to episode ends
                episode_ends = self.episode_ends
                if is_zarr:
                    episode_ends.resize(episode_ends.shape[0] + 1)
                else:
                    episode_ends.resize(episode_ends.shape[0] + 1, refcheck=False)
                episode_ends[-1] = new_len

                # rechunk
                if is_zarr:
                    if episode_ends.chunks[0] < episode_ends.shape[0]:
                        rechunk_recompress_array(self.meta, 'episode_ends', 
                            chunk_length=int(episode_ends.shape[0] * 1.5))

    def add_episode(self, 
            data: Dict[str, np.ndarray], 
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict()):
        assert(len(data) > 0)
        is_zarr = (self.backend == 'zarr')

        curr_len = self.n_steps
        episode_length = None
        for key, value in data.items():
            assert(len(value.shape) >= 1)
            if episode_length is None:
                episode_length = len(value)
            else:
                assert(episode_length == len(value))
        new_len = curr_len + episode_length

        for key, value in data.items():
            new_shape = (new_len,) + value.shape[1:]
            # create array
            if key not in self.data:
                if is_zarr:
                    cks = self._resolve_array_chunks(
                        chunks=chunks, key=key, array=value)
                    cpr = self._resolve_array_compressor(
                        compressors=compressors, key=key, array=value)
                    arr = self.data.zeros(name=key, 
                        shape=new_shape, 
                        chunks=cks,
                        dtype=value.dtype,
                        compressor=cpr)
                else:
                    # copy data to prevent modify
                    arr = np.zeros(shape=new_shape, dtype=value.dtype)
                    self.data[key] = arr
            else:
                arr = self.data[key]
                assert(value.shape[1:] == arr.shape[1:])
                # same method for both zarr and numpy
                if is_zarr:
                    arr.resize(new_shape)
                else:
                    arr.resize(new_shape, refcheck=False)
            # copy data
            arr[-value.shape[0]:] = value
        
        # append to episode ends
        episode_ends = self.episode_ends
        if is_zarr:
            episode_ends.resize(episode_ends.shape[0] + 1)
        else:
            episode_ends.resize(episode_ends.shape[0] + 1, refcheck=False)
        episode_ends[-1] = new_len

        # rechunk
        if is_zarr:
            if episode_ends.chunks[0] < episode_ends.shape[0]:
                rechunk_recompress_array(self.meta, 'episode_ends', 
                    chunk_length=int(episode_ends.shape[0] * 1.5))
    
    def drop_episode(self):
        is_zarr = (self.backend == 'zarr')
        episode_ends = self.episode_ends[:].copy()
        assert(len(episode_ends) > 0)
        start_idx = 0
        if len(episode_ends) > 1:
            start_idx = episode_ends[-2]
        for key, value in self.data.items():
            new_shape = (start_idx,) + value.shape[1:]
            if is_zarr:
                value.resize(new_shape)
            else:
                value.resize(new_shape, refcheck=False)
        if is_zarr:
            self.episode_ends.resize(len(episode_ends)-1)
        else:
            self.episode_ends.resize(len(episode_ends)-1, refcheck=False)

    def drop_episode_by_index(self, idx: int):
        """
        Drop a specific episode by index.
        
        Args:
            idx: Index of the episode to drop (0-indexed). Negative indices are supported.
        """
        is_zarr = (self.backend == 'zarr')
        episode_ends = self.episode_ends[:].copy()
        n_episodes = len(episode_ends)
        assert(n_episodes > 0), "No episodes to drop"
        
        # Handle negative indices
        if idx < 0:
            idx = n_episodes + idx
        assert(0 <= idx < n_episodes), f"Episode index {idx} out of range [0, {n_episodes-1}]"
        
        # Get the start and end indices of the episode to drop
        start_idx = 0 if idx == 0 else episode_ends[idx - 1]
        end_idx = episode_ends[idx]
        episode_length = end_idx - start_idx
        
        # For each data array, remove the episode's data by shifting subsequent data
        for key, value in self.data.items():
            current_len = value.shape[0]
            new_len = current_len - episode_length
            
            if end_idx < current_len:
                # Shift data after the dropped episode
                value[start_idx:new_len] = value[end_idx:current_len]
            
            # Resize the array
            new_shape = (new_len,) + value.shape[1:]
            if is_zarr:
                value.resize(new_shape)
            else:
                value.resize(new_shape, refcheck=False)
        
        # Update episode_ends: remove the dropped episode and adjust subsequent indices
        new_episode_ends = np.concatenate([
            episode_ends[:idx],
            episode_ends[idx + 1:] - episode_length
        ])
        
        # Resize and update episode_ends
        if is_zarr:
            self.episode_ends.resize(len(new_episode_ends))
        else:
            self.episode_ends.resize(len(new_episode_ends), refcheck=False)
        self.episode_ends[:] = new_episode_ends

    def stabilize(self):
        is_zarr = (self.backend == 'zarr')
        episode_ends = self.episode_ends[:].copy()
        if len(episode_ends) == 0:
            return
        
        # Check if all data arrays have the same length
        lengths = {key: value.shape[0] for key, value in self.data.items()}
        if len(lengths) == 0:
            return
            
        unique_lengths = set(lengths.values())
        final_len = episode_ends[-1]
        
        if len(unique_lengths) > 1:
            # Arrays have different lengths - one didn't save properly
            # Revert to the previous episode
            print(f"Warning: Data arrays have inconsistent lengths: {lengths}")
            print("Reverting to previous episode...")
            
            if len(episode_ends) > 1:
                # Revert to the end of the previous episode
                prev_len = episode_ends[-2]
                # Resize episode_ends to remove the last episode
                if is_zarr:
                    self.episode_ends.resize(len(episode_ends) - 1)
                else:
                    self.episode_ends.resize(len(episode_ends) - 1, refcheck=False)
                final_len = prev_len
            else:
                # Only one episode exists, revert to empty state
                if is_zarr:
                    self.episode_ends.resize(0)
                else:
                    self.episode_ends.resize(0, refcheck=False)
                final_len = 0
            
            print(f"Reverted to length {final_len}")
        
        # Resize all arrays to the final length
        for key, value in self.data.items():
            if value.shape[0] != final_len:
                new_shape = (final_len,) + value.shape[1:]
                if is_zarr:
                    value.resize(new_shape)
                else:
                    value.resize(new_shape, refcheck=False)
    
    def pop_episode(self):
        assert(self.n_episodes > 0)
        episode = self.get_episode(self.n_episodes-1, copy=True)
        self.drop_episode()
        return episode

    def extend(self, data):
        self.add_episode(data)

    def get_episode(self, idx, copy=False):
        idx = list(range(len(self.episode_ends)))[idx]
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx-1]
        end_idx = self.episode_ends[idx]
        result = self.get_steps_slice(start_idx, end_idx, copy=copy)
        return result
    
    def get_episode_slice(self, idx):
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx-1]
        end_idx = self.episode_ends[idx]
        return slice(start_idx, end_idx)

    def get_steps_slice(self, start, stop, step=None, copy=False):
        _slice = slice(start, stop, step)

        result = dict()
        for key, value in self.data.items():
            x = value[_slice]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            result[key] = x
        return result
    
    # =========== chunking =============
    def get_chunks(self) -> dict:
        assert self.backend == 'zarr'
        chunks = dict()
        for key, value in self.data.items():
            chunks[key] = value.chunks
        return chunks
    
    def set_chunks(self, chunks: dict):
        assert self.backend == 'zarr'
        for key, value in chunks.items():
            if key in self.data:
                arr = self.data[key]
                if value != arr.chunks:
                    check_chunks_compatible(chunks=value, shape=arr.shape)
                    rechunk_recompress_array(self.data, key, chunks=value)

    def get_compressors(self) -> dict:
        assert self.backend == 'zarr'
        compressors = dict()
        for key, value in self.data.items():
            compressors[key] = value.compressor
        return compressors

    def set_compressors(self, compressors: dict):
        assert self.backend == 'zarr'
        for key, value in compressors.items():
            if key in self.data:
                arr = self.data[key]
                compressor = self.resolve_compressor(value)
                if compressor != arr.compressor:
                    rechunk_recompress_array(self.data, key, compressor=compressor)



def match_observations_to_actions(action_timestamps, cam_timestamps):
    """
    Returns indices of actions corresponding to the closest smaller 
    (or equal) timestamp for each camera observation.
    """
    # 1. Find insertion points
    # side='right' finds the first index where the value is strictly greater 
    # than the query.
    indices = np.searchsorted(action_timestamps, cam_timestamps, side='right')
    
    # 2. Shift left to get the "closest smaller/equal"
    # Since searchsorted returns the index *after* the match, we subtract 1
    matched_indices = indices - 1
    
    # 3. (Optional but recommended) Handle boundary cases
    # If a camera obs happened before the first action, index will be -1.
    # You might want to filter those out.
    valid_mask = matched_indices >= 0
    
    return matched_indices[valid_mask], valid_mask

def align_camera_timestamps(cameras_data):
    """
    Align camera timestamps using the camera with the least frames as baseline.
    Returns aligned camera data where all cameras have the same number of observations.
    """
    if not cameras_data:
        return {}
    
    # Find camera with minimum number of frames
    min_frames = float('inf')
    latest_start = -float('inf')
    baseline_serial = None
    for serial, data in cameras_data.items():
        n_frames = len(data['cam_timestamps'])
        if n_frames <= min_frames and data['cam_timestamps'][0] >= latest_start:
            min_frames = n_frames
            baseline_serial = serial
            latest_start = data['cam_timestamps'][0]
    
    print(f"Using camera {baseline_serial} as baseline with {min_frames} frames")
    baseline_timestamps = cameras_data[baseline_serial]['cam_timestamps']
    
    # Align all cameras to baseline timestamps
    aligned_cameras = {}
    for i, (serial, data) in enumerate(cameras_data.items()):
        # Create aligned data with camera index naming
        aligned_data = {}
        
        if serial == baseline_serial:
            # Baseline camera - just rename keys with index
            for key, value in data.items():
                if key == 'cam_timestamps':
                    aligned_data[key] = value
                elif key == 'color':
                    aligned_data[f'camera{i}_rgb'] = value
                elif key == 'depth':
                    aligned_data[f'camera{i}_depth'] = value
                else:
                    aligned_data[f'camera{i}_{key}'] = value
        else:
            # Other cameras - find closest matches to each baseline timestamp
            # Use searchsorted to find closest timestamps in the other camera
            indices = np.searchsorted(data['cam_timestamps'], baseline_timestamps, side='left')
            # Clamp indices to valid range
            indices = np.clip(indices, 0, len(data['cam_timestamps']) - 1)
            
            for key, value in data.items():
                if key == 'cam_timestamps':
                    aligned_data[key] = baseline_timestamps  # Use exact baseline timestamps
                elif key == 'color':
                    aligned_data[f'camera{i}_rgb'] = value[indices]
                elif key == 'depth':
                    aligned_data[f'camera{i}_depth'] = value[indices]
                else:
                    aligned_data[f'camera{i}_{key}'] = value[indices]
        
        aligned_cameras[serial] = aligned_data
    
    # All cameras should now have exactly min_frames length (same as baseline)
    # Merge all camera data into single dictionary
    merged_data = {'cam_timestamps': baseline_timestamps}
    for serial, data in aligned_cameras.items():
        for key, value in data.items():
            if key != 'cam_timestamps':  # Don't duplicate timestamps
                merged_data[key] = value
    
    return merged_data



def preprocess_robot_actions(robot_data, timestamps_key='action_timestamps'):
    """
    Combine robot action keys into a single 'action' array.
    Format: [joint_pos_L, gripper_pos_L, joint_pos_R, gripper_pos_R]
    """
    # Extract individual components
    joint_pos_L = robot_data['joint_pos_L']  # Shape: (N, dof)
    gripper_pos_L = robot_data['gripper_pos_L'].reshape(-1, 1)  # Shape: (N, 1)
    joint_pos_R = robot_data['joint_pos_R']  # Shape: (N, dof)
    gripper_pos_R = robot_data['gripper_pos_R'].reshape(-1, 1)  # Shape: (N, 1)
    
    # Concatenate all components
    action = np.concatenate([
        joint_pos_L,
        gripper_pos_L,
        joint_pos_R,
        gripper_pos_R
    ], axis=1)
    
    # Create new data dictionary with action and timestamps only
    if timestamps_key is not None:
        processed_data = {
            'action': action,
            'action_timestamps': robot_data[timestamps_key]
        }
    else:
        processed_data = {
            'action': action
        }
    
    return processed_data


# =============================================================================
# External Force Detection and Torque Visualization
# =============================================================================

class ExternalForceDetector:
    """Calibration-based external force detector with full history tracking."""
    
    def __init__(self, n_dims, threshold_sigma=3.0):
        self.n_dims = n_dims
        self.threshold = threshold_sigma
        
        # Calibration state
        self.is_calibrated = False
        self.calibration_samples = [[] for _ in range(n_dims)]
        self.calibrated_mean = None
        self.calibrated_std = None
        
        # Full history for plotting
        self.full_torque_history = [[] for _ in range(n_dims)]
        self.full_contact_history = [[] for _ in range(n_dims)]
    
    def update(self, tau_ext):
        """Returns bool array: True if contact detected per dimension."""
        contact = []
        
        if not self.is_calibrated:
            # Calibration mode: just collect samples, return all False
            for i, val in enumerate(tau_ext):
                self.calibration_samples[i].append(val)
                contact.append(False)
                # Store in full history
                self.full_torque_history[i].append(val)
                self.full_contact_history[i].append(False)
        else:
            # Eval mode: compare against fixed calibrated baseline
            for i, val in enumerate(tau_ext):
                z = abs(val - self.calibrated_mean[i]) / self.calibrated_std[i]
                is_contact = z > self.threshold
                contact.append(is_contact)
                # Store in full history
                self.full_torque_history[i].append(val)
                self.full_contact_history[i].append(is_contact)
        
        return np.array(contact)
    
    def finish_calibration(self):
        """Lock in baseline mean/std from calibration samples."""
        self.calibrated_mean = np.array([np.mean(s) for s in self.calibration_samples])
        self.calibrated_std = np.array([max(np.std(s), 0.01) for s in self.calibration_samples])
        self.is_calibrated = True
        return self.calibrated_mean, self.calibrated_std
    
    def clear_history(self):
        """Clear all history for new recording."""
        self.is_calibrated = False
        self.calibration_samples = [[] for _ in range(self.n_dims)]
        self.calibrated_mean = None
        self.calibrated_std = None
        self.full_torque_history = [[] for _ in range(self.n_dims)]
        self.full_contact_history = [[] for _ in range(self.n_dims)]
    
    def get_history_arrays(self):
        """Return history as numpy arrays."""
        torques = [np.array(h) for h in self.full_torque_history]
        contacts = [np.array(h) for h in self.full_contact_history]
        return torques, contacts


def plot_external_torque_history(detector_L, detector_R, save_path=None):
    """Plot external torque history for both arms with contact detection markers and binary plots."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    torques_L, contacts_L = detector_L.get_history_arrays()
    torques_R, contacts_R = detector_R.get_history_arrays()
    
    # Get calibration data
    mean_L, std_L = detector_L.calibrated_mean, detector_L.calibrated_std
    mean_R, std_R = detector_R.calibrated_mean, detector_R.calibrated_std
    threshold = detector_L.threshold  # Same for both
    
    n_dims = detector_L.n_dims
    joint_names = [f"Joint {i+1}" for i in range(n_dims - 1)] + ["Gripper"]
    
    # Create figure with custom grid: each joint gets a torque plot (3 units) + binary plot (1 unit)
    fig = plt.figure(figsize=(14, 3.0 * n_dims))
    gs = GridSpec(n_dims * 2, 2, figure=fig, height_ratios=[3, 1] * n_dims, hspace=0.1, wspace=0.25)
    fig.suptitle("External Torque History with Contact Detection", fontsize=14, y=0.995)
    
    for i in range(n_dims):
        # Left arm - torque plot
        ax_L = fig.add_subplot(gs[i * 2, 0])
        t = np.arange(len(torques_L[i]))
        
        # Compute z-scores for coloring
        if mean_L is not None:
            zscore_L = np.abs(torques_L[i] - mean_L[i]) / std_L[i]
            # Color segment green if either endpoint exceeds threshold
            for j in range(len(t) - 1):
                is_contact = (zscore_L[j] > threshold) or (zscore_L[j+1] > threshold)
                color = 'green' if is_contact else 'red'
                ax_L.plot(t[j:j+2], torques_L[i][j:j+2], color=color, linewidth=0.8, zorder=2)
        else:
            ax_L.plot(t, torques_L[i], 'r-', linewidth=0.8, zorder=2)
        
        # Add mean and threshold lines
        if mean_L is not None:
            ax_L.axhline(mean_L[i], color='lightblue', linewidth=1.5, linestyle='-', label='mean', zorder=1)
            ax_L.axhline(mean_L[i] + threshold * std_L[i], color='blue', linewidth=0.8, linestyle='--', label=f'±{threshold}σ', zorder=1)
            ax_L.axhline(mean_L[i] - threshold * std_L[i], color='blue', linewidth=0.8, linestyle='--', zorder=1)
        
        ax_L.set_ylabel("Torque (Nm)")
        ax_L.set_title(f"L {joint_names[i]}")
        ax_L.grid(True, alpha=0.3)
        ax_L.set_xticklabels([])  # Hide x labels, binary plot below shares x-axis
        if i == 0:
            ax_L.legend(loc='upper right', fontsize=8)
        
        # Left arm - binary contact plot
        ax_L_bin = fig.add_subplot(gs[i * 2 + 1, 0], sharex=ax_L)
        contact_binary_L = contacts_L[i].astype(int)
        ax_L_bin.fill_between(t, 0, contact_binary_L, step='post', alpha=0.7, color='green')
        ax_L_bin.set_ylim(-0.1, 1.1)
        ax_L_bin.set_yticks([0, 1])
        ax_L_bin.set_yticklabels(['0', '1'], fontsize=8)
        ax_L_bin.set_ylabel("Contact", fontsize=8)
        ax_L_bin.grid(True, alpha=0.3)
        if i < n_dims - 1:
            ax_L_bin.set_xticklabels([])
        
        # Right arm - torque plot
        ax_R = fig.add_subplot(gs[i * 2, 1])
        t = np.arange(len(torques_R[i]))
        
        # Compute z-scores for coloring
        if mean_R is not None:
            zscore_R = np.abs(torques_R[i] - mean_R[i]) / std_R[i]
            # Color segment green if either endpoint exceeds threshold
            for j in range(len(t) - 1):
                is_contact = (zscore_R[j] > threshold) or (zscore_R[j+1] > threshold)
                color = 'green' if is_contact else 'red'
                ax_R.plot(t[j:j+2], torques_R[i][j:j+2], color=color, linewidth=0.8, zorder=2)
        else:
            ax_R.plot(t, torques_R[i], 'r-', linewidth=0.8, zorder=2)
        
        # Add mean and threshold lines
        if mean_R is not None:
            ax_R.axhline(mean_R[i], color='lightblue', linewidth=1.5, linestyle='-', label='mean', zorder=1)
            ax_R.axhline(mean_R[i] + threshold * std_R[i], color='blue', linewidth=0.8, linestyle='--', label=f'±{threshold}σ', zorder=1)
            ax_R.axhline(mean_R[i] - threshold * std_R[i], color='blue', linewidth=0.8, linestyle='--', zorder=1)
        
        ax_R.set_ylabel("Torque (Nm)")
        ax_R.set_title(f"R {joint_names[i]}")
        ax_R.grid(True, alpha=0.3)
        ax_R.set_xticklabels([])
        if i == 0:
            ax_R.legend(loc='upper right', fontsize=8)
        
        # Right arm - binary contact plot
        ax_R_bin = fig.add_subplot(gs[i * 2 + 1, 1], sharex=ax_R)
        contact_binary_R = contacts_R[i].astype(int)
        ax_R_bin.fill_between(t, 0, contact_binary_R, step='post', alpha=0.7, color='green')
        ax_R_bin.set_ylim(-0.1, 1.1)
        ax_R_bin.set_yticks([0, 1])
        ax_R_bin.set_yticklabels(['0', '1'], fontsize=8)
        ax_R_bin.set_ylabel("Contact", fontsize=8)
        ax_R_bin.grid(True, alpha=0.3)
        if i < n_dims - 1:
            ax_R_bin.set_xticklabels([])
    
    # X-axis label for bottom plots
    fig.text(0.27, 0.02, "Timestep", ha='center', fontsize=10)
    fig.text(0.73, 0.02, "Timestep", ha='center', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved torque history to {save_path}")
    
    plt.close()


def plot_zscore_history(detector_L, detector_R, save_path=None):
    """Plot Z-score ((value - mean) / std) history for both arms with threshold lines."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    torques_L, contacts_L = detector_L.get_history_arrays()
    torques_R, contacts_R = detector_R.get_history_arrays()
    
    # Get calibration data
    mean_L, std_L = detector_L.calibrated_mean, detector_L.calibrated_std
    mean_R, std_R = detector_R.calibrated_mean, detector_R.calibrated_std
    threshold = detector_L.threshold
    
    if mean_L is None or mean_R is None:
        print("[Plot] Cannot plot Z-scores: calibration data not available")
        return
    
    n_dims = detector_L.n_dims
    joint_names = [f"Joint {i+1}" for i in range(n_dims - 1)] + ["Gripper"]
    
    # Create figure with custom grid: each joint gets a zscore plot (3 units) + binary plot (1 unit)
    fig = plt.figure(figsize=(14, 3.0 * n_dims))
    gs = GridSpec(n_dims * 2, 2, figure=fig, height_ratios=[3, 1] * n_dims, hspace=0.1, wspace=0.25)
    fig.suptitle("Z-Score History: (τ - mean) / std", fontsize=14, y=0.995)
    
    for i in range(n_dims):
        # Compute Z-scores
        zscore_L = (torques_L[i] - mean_L[i]) / std_L[i]
        zscore_R = (torques_R[i] - mean_R[i]) / std_R[i]
        
        # Left arm - Z-score plot
        ax_L = fig.add_subplot(gs[i * 2, 0])
        t = np.arange(len(zscore_L))
        
        # Color segment green if either endpoint exceeds threshold
        for j in range(len(t) - 1):
            is_contact = (np.abs(zscore_L[j]) > threshold) or (np.abs(zscore_L[j+1]) > threshold)
            color = 'green' if is_contact else 'red'
            ax_L.plot(t[j:j+2], zscore_L[j:j+2], color=color, linewidth=0.8, zorder=2)
        
        # Add mean (0) and threshold lines
        ax_L.axhline(0, color='blue', linewidth=0.8, linestyle='-', label='mean (0)', zorder=1)
        ax_L.axhline(threshold, color='blue', linewidth=1.2, linestyle='--', label=f'±{threshold}σ', zorder=1)
        ax_L.axhline(-threshold, color='blue', linewidth=1.2, linestyle='--', zorder=1)
        
        ax_L.set_ylabel("Z-score (σ)")
        ax_L.set_title(f"L {joint_names[i]}")
        ax_L.grid(True, alpha=0.3)
        ax_L.set_xticklabels([])
        if i == 0:
            ax_L.legend(loc='upper right', fontsize=8)
        
        # Left arm - binary contact plot
        ax_L_bin = fig.add_subplot(gs[i * 2 + 1, 0], sharex=ax_L)
        contact_binary_L = contacts_L[i].astype(int)
        ax_L_bin.fill_between(t, 0, contact_binary_L, step='post', alpha=0.7, color='green')
        ax_L_bin.set_ylim(-0.1, 1.1)
        ax_L_bin.set_yticks([0, 1])
        ax_L_bin.set_yticklabels(['0', '1'], fontsize=8)
        ax_L_bin.set_ylabel("Contact", fontsize=8)
        ax_L_bin.grid(True, alpha=0.3)
        if i < n_dims - 1:
            ax_L_bin.set_xticklabels([])
        
        # Right arm - Z-score plot
        ax_R = fig.add_subplot(gs[i * 2, 1])
        t = np.arange(len(zscore_R))
        
        # Color segment green if either endpoint exceeds threshold
        for j in range(len(t) - 1):
            is_contact = (np.abs(zscore_R[j]) > threshold) or (np.abs(zscore_R[j+1]) > threshold)
            color = 'green' if is_contact else 'red'
            ax_R.plot(t[j:j+2], zscore_R[j:j+2], color=color, linewidth=0.8, zorder=2)
        
        # Add mean (0) and threshold lines
        ax_R.axhline(0, color='blue', linewidth=0.8, linestyle='-', label='mean (0)', zorder=1)
        ax_R.axhline(threshold, color='blue', linewidth=1.2, linestyle='--', label=f'±{threshold}σ', zorder=1)
        ax_R.axhline(-threshold, color='blue', linewidth=1.2, linestyle='--', zorder=1)
        
        ax_R.set_ylabel("Z-score (σ)")
        ax_R.set_title(f"R {joint_names[i]}")
        ax_R.grid(True, alpha=0.3)
        ax_R.set_xticklabels([])
        if i == 0:
            ax_R.legend(loc='upper right', fontsize=8)
        
        # Right arm - binary contact plot
        ax_R_bin = fig.add_subplot(gs[i * 2 + 1, 1], sharex=ax_R)
        contact_binary_R = contacts_R[i].astype(int)
        ax_R_bin.fill_between(t, 0, contact_binary_R, step='post', alpha=0.7, color='green')
        ax_R_bin.set_ylim(-0.1, 1.1)
        ax_R_bin.set_yticks([0, 1])
        ax_R_bin.set_yticklabels(['0', '1'], fontsize=8)
        ax_R_bin.set_ylabel("Contact", fontsize=8)
        ax_R_bin.grid(True, alpha=0.3)
        if i < n_dims - 1:
            ax_R_bin.set_xticklabels([])
    
    # X-axis label for bottom plots
    fig.text(0.27, 0.02, "Timestep", ha='center', fontsize=10)
    fig.text(0.73, 0.02, "Timestep", ha='center', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved Z-score history to {save_path}")
    
    plt.close()


def plot_tanh_normalized_history(detector_L, detector_R, save_path=None):
    """Plot tanh-normalized torque history: tanh((τ - mean) / std / threshold_sigma)."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    torques_L, contacts_L = detector_L.get_history_arrays()
    torques_R, contacts_R = detector_R.get_history_arrays()
    
    # Get calibration data
    mean_L, std_L = detector_L.calibrated_mean, detector_L.calibrated_std
    mean_R, std_R = detector_R.calibrated_mean, detector_R.calibrated_std
    threshold = detector_L.threshold  # Use threshold_sigma as scaling factor
    
    if mean_L is None or mean_R is None:
        print("[Plot] Cannot plot tanh-normalized: calibration data not available")
        return
    
    # Threshold in tanh space: tanh(1) ≈ 0.76
    tanh_threshold = np.tanh(1.0)
    
    n_dims = detector_L.n_dims
    joint_names = [f"Joint {i+1}" for i in range(n_dims - 1)] + ["Gripper"]
    
    # Create figure with custom grid: each joint gets a tanh plot (3 units) + binary plot (1 unit)
    fig = plt.figure(figsize=(14, 3.0 * n_dims))
    gs = GridSpec(n_dims * 2, 2, figure=fig, height_ratios=[3, 1] * n_dims, hspace=0.1, wspace=0.25)
    fig.suptitle(f"Tanh-Normalized Torque: tanh((τ - mean) / std / {threshold})", fontsize=14, y=0.995)
    
    for i in range(n_dims):
        # Compute tanh-normalized values
        tanh_L = np.tanh((torques_L[i] - mean_L[i]) / std_L[i] / threshold)
        tanh_R = np.tanh((torques_R[i] - mean_R[i]) / std_R[i] / threshold)
        
        # Left arm - tanh plot
        ax_L = fig.add_subplot(gs[i * 2, 0])
        t = np.arange(len(tanh_L))
        
        # Color segment green if either endpoint exceeds threshold (in tanh space)
        for j in range(len(t) - 1):
            is_contact = (np.abs(tanh_L[j]) > tanh_threshold) or (np.abs(tanh_L[j+1]) > tanh_threshold)
            color = 'green' if is_contact else 'red'
            ax_L.plot(t[j:j+2], tanh_L[j:j+2], color=color, linewidth=0.8, zorder=2)
        
        # Add threshold lines
        ax_L.axhline(0, color='blue', linewidth=0.8, linestyle='-', label='mean (0)', zorder=1)
        ax_L.axhline(tanh_threshold, color='blue', linewidth=1.2, linestyle='--', label=f'±tanh(1)≈{tanh_threshold:.2f}', zorder=1)
        ax_L.axhline(-tanh_threshold, color='blue', linewidth=1.2, linestyle='--', zorder=1)
        ax_L.axhline(1.0, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
        ax_L.axhline(-1.0, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
        
        ax_L.set_ylim(-1.1, 1.1)
        ax_L.set_ylabel("tanh(z/σ)")
        ax_L.set_title(f"L {joint_names[i]}")
        ax_L.grid(True, alpha=0.3)
        ax_L.set_xticklabels([])
        if i == 0:
            ax_L.legend(loc='upper right', fontsize=8)
        
        # Left arm - binary contact plot
        ax_L_bin = fig.add_subplot(gs[i * 2 + 1, 0], sharex=ax_L)
        contact_binary_L = contacts_L[i].astype(int)
        ax_L_bin.fill_between(t, 0, contact_binary_L, step='post', alpha=0.7, color='green')
        ax_L_bin.set_ylim(-0.1, 1.1)
        ax_L_bin.set_yticks([0, 1])
        ax_L_bin.set_yticklabels(['0', '1'], fontsize=8)
        ax_L_bin.set_ylabel("Contact", fontsize=8)
        ax_L_bin.grid(True, alpha=0.3)
        if i < n_dims - 1:
            ax_L_bin.set_xticklabels([])
        
        # Right arm - tanh plot
        ax_R = fig.add_subplot(gs[i * 2, 1])
        t = np.arange(len(tanh_R))
        
        # Color segment green if either endpoint exceeds threshold (in tanh space)
        for j in range(len(t) - 1):
            is_contact = (np.abs(tanh_R[j]) > tanh_threshold) or (np.abs(tanh_R[j+1]) > tanh_threshold)
            color = 'green' if is_contact else 'red'
            ax_R.plot(t[j:j+2], tanh_R[j:j+2], color=color, linewidth=0.8, zorder=2)
        
        # Add threshold lines
        ax_R.axhline(0, color='blue', linewidth=0.8, linestyle='-', label='mean (0)', zorder=1)
        ax_R.axhline(tanh_threshold, color='blue', linewidth=1.2, linestyle='--', label=f'±tanh(1)≈{tanh_threshold:.2f}', zorder=1)
        ax_R.axhline(-tanh_threshold, color='blue', linewidth=1.2, linestyle='--', zorder=1)
        ax_R.axhline(1.0, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
        ax_R.axhline(-1.0, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
        
        ax_R.set_ylim(-1.1, 1.1)
        ax_R.set_ylabel("tanh(z/σ)")
        ax_R.set_title(f"R {joint_names[i]}")
        ax_R.grid(True, alpha=0.3)
        ax_R.set_xticklabels([])
        if i == 0:
            ax_R.legend(loc='upper right', fontsize=8)
        
        # Right arm - binary contact plot
        ax_R_bin = fig.add_subplot(gs[i * 2 + 1, 1], sharex=ax_R)
        contact_binary_R = contacts_R[i].astype(int)
        ax_R_bin.fill_between(t, 0, contact_binary_R, step='post', alpha=0.7, color='green')
        ax_R_bin.set_ylim(-0.1, 1.1)
        ax_R_bin.set_yticks([0, 1])
        ax_R_bin.set_yticklabels(['0', '1'], fontsize=8)
        ax_R_bin.set_ylabel("Contact", fontsize=8)
        ax_R_bin.grid(True, alpha=0.3)
        if i < n_dims - 1:
            ax_R_bin.set_xticklabels([])
    
    # X-axis label for bottom plots
    fig.text(0.27, 0.02, "Timestep", ha='center', fontsize=10)
    fig.text(0.73, 0.02, "Timestep", ha='center', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved tanh-normalized history to {save_path}")
    
    plt.close()


class TimestampPlotter:
    """
    Visualizes timestamp alignment on a single horizontal timeline.
    All cameras and poll times on one line, scrollable horizontally.
    """
    
    def __init__(self):
        self.poll_data = []  # List of poll info dicts
        self.all_serials = set()
    
    def record_poll(self, camera_data, ref_serial, ref_timestamp, poll_time):
        """
        Record data from one poll.
        
        Args:
            camera_data: dict of serial -> {hw_timestamp: array, ...}
            ref_serial: which camera was chosen as best
            ref_timestamp: the reference timestamp used (from best camera)
            poll_time: time.time() when poll happened
        """
        poll_info = {
            'poll_time': poll_time,
            'ref_serial': ref_serial,
            'ref_timestamp': ref_timestamp,
            'cameras': {}  # serial -> hw_timestamp of newest frame
        }
        
        for serial, data in camera_data.items():
            self.all_serials.add(serial)
            # Get the newest frame's hw_timestamp
            if len(data['hw_timestamp']) > 0:
                poll_info['cameras'][serial] = data['hw_timestamp'][-1]
        
        self.poll_data.append(poll_info)
    
    def save_plot(self, output_path):
        """Save a horizontally scrollable single-line timeline."""
        if not self.poll_data:
            print("[TimestampPlotter] No data to plot")
            return
        
        n_polls = len(self.poll_data)
        all_serials = sorted(self.all_serials)
        n_cameras = len(all_serials)
        
        print(f"[TimestampPlotter] Plotting {n_polls} polls, {n_cameras} cameras")
        
        # Get time range
        first_poll_time = self.poll_data[0]['poll_time']
        
        # Debug: check poll spacing
        if n_polls > 1:
            poll_intervals = []
            for i in range(1, min(20, n_polls)):  # Check first 20 intervals
                interval_ms = (self.poll_data[i]['poll_time'] - self.poll_data[i-1]['poll_time']) * 1000
                poll_intervals.append(interval_ms)
            avg_interval = np.mean(poll_intervals)
            std_interval = np.std(poll_intervals)
            print(f"[TimestampPlotter] Poll intervals: avg={avg_interval:.2f}ms, std={std_interval:.2f}ms, range=[{min(poll_intervals):.1f}, {max(poll_intervals):.1f}]ms")
        
        # Calculate and print camera FPS based on hw_timestamp
        if n_polls > 1:
            print("[TimestampPlotter] Camera FPS (based on hw_timestamp):")
            for serial in all_serials:
                cam_intervals = []
                prev_ts = None
                for poll in self.poll_data:
                    if serial in poll['cameras']:
                        ts = poll['cameras'][serial]
                        if prev_ts is not None and ts != prev_ts:  # Only count if timestamp changed
                            interval_ms = (ts - prev_ts) * 1000
                            if interval_ms > 0:
                                cam_intervals.append(interval_ms)
                        prev_ts = ts
                if cam_intervals:
                    avg_interval_ms = np.mean(cam_intervals)
                    fps = 1000.0 / avg_interval_ms
                    std_interval_ms = np.std(cam_intervals)
                    print(f"  Camera {serial[-4:]}: {fps:.2f} fps (interval: {avg_interval_ms:.2f}ms ± {std_interval_ms:.2f}ms)")
        
        # Figure width: ~5 polls per 20 inches = 4 inches per poll
        # Cap at 200 inches max to avoid X11/memory issues
        fig_width = min(200, max(20, n_polls * 4))
        fig_height = 4
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Assign colors to each camera (using distinct colors)
        cmap = plt.cm.tab10
        serial_colors = {serial: cmap(i % 10) for i, serial in enumerate(all_serials)}
        
        # Y positions: stagger cameras slightly so they don't overlap
        # Poll ticks at y=0, cameras spread from y=0.3 to y=0.8
        y_poll = 0
        y_offsets = {serial: 0.3 + 0.5 * i / max(1, n_cameras - 1) 
                    for i, serial in enumerate(all_serials)} if n_cameras > 1 else {all_serials[0]: 0.5}
        
        # Plot each poll
        for poll_idx, poll in enumerate(self.poll_data):
            # Time relative to first poll (in ms)
            poll_rel_time = (poll['poll_time'] - first_poll_time) * 1000
            
            # Draw poll tick (black triangle pointing up at y=0)
            ax.scatter(poll_rel_time, y_poll, marker='^', color='black', s=100, zorder=10)
            
            # Draw each camera's timestamp for this poll
            for serial, cam_ts in poll['cameras'].items():
                cam_rel_time = (cam_ts - first_poll_time) * 1000
                y_cam = y_offsets[serial]
                color = serial_colors[serial]
                
                # Mark if this is the chosen "best" camera
                if serial == poll['ref_serial']:
                    marker = 'D'  # Diamond
                    size = 120
                    edgecolor = 'red'
                    linewidth = 2.5
                else:
                    marker = 'o'
                    size = 60
                    edgecolor = 'black'
                    linewidth = 0.5
                
                ax.scatter(cam_rel_time, y_cam, marker=marker, color=color, 
                          s=size, edgecolors=edgecolor, linewidths=linewidth, zorder=5)
                
                # Draw faint line from poll tick to camera timestamp
                ax.plot([poll_rel_time, cam_rel_time], [y_poll, y_cam], 
                       color=color, alpha=0.3, linewidth=1, zorder=1)
        
        # Formatting
        ax.set_ylim(-0.2, 1.0)
        ax.set_yticks([])  # No y-axis ticks needed for single timeline
        ax.set_xlabel('Time (ms, relative to first poll)', fontsize=12)
        ax.set_title(f'Timestamp Alignment Timeline: {n_polls} polls, {n_cameras} cameras\n'
                    f'▲ = Poll time, ◆ (red edge) = chosen camera, ● = other cameras',
                    fontsize=12)
        ax.axhline(y=0, color='black', alpha=0.3, linewidth=1)  # Timeline base
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        
        # Legend for cameras
        legend_elements = [
            plt.Line2D([0], [0], marker='^', color='black', linestyle='None', 
                      markersize=10, label='Poll Time'),
        ]
        for serial in all_serials:
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color=serial_colors[serial], 
                          linestyle='None', markersize=8, label=f'Cam {serial[-4:]}')
            )
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9, ncol=2)
        
        plt.tight_layout()
        
        # Save as PDF for better scrollability, with PNG fallback
        pdf_path = output_path.replace('.png', '.pdf')
        try:
            plt.savefig(pdf_path, dpi=100, bbox_inches='tight')
            print(f"[TimestampPlotter] Saved plot to {pdf_path}")
        except Exception as e:
            print(f"[TimestampPlotter] PDF save failed: {e}")
        
        # Also save a smaller PNG for quick preview
        try:
            plt.savefig(output_path, dpi=50, bbox_inches='tight')
            print(f"[TimestampPlotter] Saved preview to {output_path}")
        except Exception as e:
            print(f"[TimestampPlotter] PNG save failed: {e}")
        
        plt.close(fig)
    
    def clear(self):
        """Clear all recorded data."""
        self.poll_data = []
        self.all_serials = set()
