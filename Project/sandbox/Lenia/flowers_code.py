# !/usr/bin/env python3
"""Script following flowers' lab source code for how to model
obstacles in the lenia environment"""

## IMPORTS ##
from addict import Dict
import torch
from copy import deepcopy
import numpy as np
import warnings
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict
from torchvision.transforms.functional import rotate
torch.set_default_tensor_type("torch.cuda.FloatTensor")

import os
os.environ["FFMPEG_BINARY"] = "ffmpeg"
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from IPython.display import HTML, display, clear_output

## PREPARATIONS
class VideoWriter:
    def __init__(self, filename, fps=30.0, **kw):
        self.writer = None
        self.params = dict(filename=filename, fps=fps, **kw)

    def add(self, img):
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size = (w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0,1)*255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)
        self.writer.write_frame(img)

    def close(self):
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()

    def show(self, **kw):
        self.close()
        fn = self.params["filename"]
        display(mvp.ipython_display(fn, **kw))

    def complex_mult_torch(X,Y):
        """Computes complex multiplication in pytorch when the tensor last dimension is 2:
        0 is the real component and 1 is the imaginary one"""
        assert X.shape[-1] == 2 and Y.shape[-1] == 2  # Last dimension must be 2
        return torch.stack(
            (X[...,0] * Y[..., 0] - X[..., 1]*Y[...,1],
             X[...,0] * Y[...,1] + X[...,1] * Y[...,0]),
            dim = -1
        )

    def roll_n(X, axis, n):
        """Rolls a tensor with a shift n on specified axis"""
        f_idx = tuple(slice(None, None, None) if i !=axis else slice(0, n, None)
                       for i in range(X.dim()))
        b_idx = tuple(slice(None, None, None) if i !=axis else slice(n, None, None)
                      for i in range(X.dim()))
        front = X[F_idx]
        back = X[b_idx]
        return torch.cat([back, front], axis)

######## DEFINE SPACE #########
class Space(object):
    """
    Defines the init_space, gnome_space and intervention_space of a system
    """
    def __init__(self, shape=None, dtype=None):
        self.shape = None if shape is None else tuple(shape)
        self.dtype = dtype

    def sample(self):
        """Randomely sample an element of this space.
        Can be uniform or non-uniform sampling based on boundedness of space"""
        raise NotImplementedError

    def mutate(self, x):
        """Randomely mutate an element of this space"""
        raise NotImplementedError

    def __contains__(self, x):
        """Return boolean specifying if x is a valid
        member of this space"""
        raise NotImplementedError

    def contains(self, x):
        """Return boolean specifying if x is a valid member of this space"""
        aise NotImplementedError

    def clamp(self, x):
        """Return a valid clamped value of x inside of space's bounds"""
        raise NotImplementedError

    def __contains__(self, x):
        return self.contains(x)

class DiscreteSpace(Space):
    """A discrete space in math: {0, 1, ..., n-1}
    Mutation is gaussian by default: please create custom space
    inheriting from discrete space for custom mutation functions

    Example::
        DiscreteSpace(2)"""

    def __init__(self, n, mutation_mean=0.0, mutation_std=1.0, indpb=1.0):
        assert n >= 0
        self.n = n

        # mutation_mean: mean for gaussian addition mutation
        # mutation_std: std for gaussian mutation addition
        # indpb: independent probability for each attribute to be mutated

        # Store each as tensor
        self.mutation_mean = torch.as_tensor(mutation_mean, dtype=torch.float64)
        self.mutation_std = torch.as_tensor(mutation_std, dtype=torch.float64)
        self.indpb = torch.as_tensor(indpb, dtype=torch.float64)
        super(DiscreteSpace, self).__init__((), torch.int64)

    def sample(self):
        return torch.randint(self.n, ())

    def mutate(self, x):
        mutate_mask = torch.rand(self.shape) < self.indpb
        noise = torch.normal(self.mutation_mean, self.mutation_std, ())
        x = x.type(torch.float64) + mutate_mask*noise
        x = torch.floor(x).type(self.dtype)
        if not self.contains(x):
            return self.clamp(x)
        else:
            return x

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif not x.dtype.is_floating_point and (x.shape==()): # integer or size 0
            as_int = int(x)
        else:
            return False
        return 0 <= as_int < self.n

    def clamp(self, x):
        x = torch.max(x, torch.as_tensor(0, dtype=self.dtype, device = x.device))
        x = torch.min(x, torch.as_tensor(self.n-1, dtype=self.dtype, device=x.device))
        return x

    def __repr__(self):
        return "DiscreteSpace(%d)" %self.n

    def __eq__(self, other):
        return isinstance(other, DiscreteSpace) and self.n == other.n

class BoxSpace(Space):
    """
    A (poissibly unbounded) box in R^n. Specifically, a box represents the
    cartesian product of n closed intervals. Each interval has the form of one
    of [a, b], (-oo, b], [a, oo), or (-oo, oo)

    There are two common use cases:
    *Identical bound for each dimension:
    >>> BoxSpace(low=-1.0, high=2.0, shape= (3, 4), dtype=torch.float32)
    box(3,4)

    * Independent bound for each dimension:
    >>> BosSpace(low=torch.tensor([-1.0, -2.0]), high=torch.tensor([2.0, 4.0]), dtype=torch.float32)
    box(2,)
    """

    def __init__(self, low, high, shape=None, dtype=torch.float32, mutation_mean=0.0,
                 mutation_std=1.0, indp=1.0):
        assert dtype is not None, 'dtype must be explicity provided.'
        self.dtype=dtype

        # Determine shape if it isn't provided directly
        if shape is not None:
            shape = tuple(shape)
            assert isinstance(low, numbers.Number) or low.shape == shape, "low.shape doesn't match provided shape"
            assert isinstance(high, numbers.Number) or high.shape == shape, "high.shape doesn't match prodivded shape"
        elif not isinstance(low, numbers.Number):
            shape.low.shape
            assert isinstance(high, numbers.Number) or high.shape == shape, "High.shape doesn't match  low.shape"
        elif not isinstance(high, numbers.Number):
            shape = high.shape
            assert isinstance(low, numbers.Number) or low.shape == shape, "Low.shape doesn't match high.shape"
        else:
            raise ValueError("Shape must be provided or inferred from shapes of low or high")

        if isinstance(low, numbers.Number):
            low = torch.full(shape, low, dtype=dtype)

        if isinstance(high, numbers.Number):
            high = torch.full(shape, high, dtype=dtype)

        self.shape = shape
        self.low = low.type(self.dtype)
        self.high = high.type(self.dtype)

        # Boolean arrays which indicate the interval for each type of coordinate
        self.bounded_below = ~torch.isneginf(self.low)
        self.bounded_above = ~torch.isposinf(self.high)

        # Mutation_mean: mean for gussian addition mutation
        # Mutation_std: std for gaussian addition mutation
        # indpb: independent probability for each attribute mutated

        if isinstance(mutation_mean, numbers.Number):
            mutation_mean = torch.full(self.shape, mutation_mean, dtype=torch.float64)
        self.mutation_mean = torch.as_tensor(mutation_mean, dtype=torch.float64)
        if isinstance(mutation_std, numbers.Number):
            mutation_std = torch.full(self.shape, mutation_std, dtype=torch.float64)
        self.mutation_std = torch.as_tensor(mutation_std, dtype=torch.float64)
        if isinstance(indpb, numbers.Number):
            indpb = torch.full(self.shape, indpb, dtype=torch.float64)
        self.indpb = torch.as_tensor(indpb, dtype-torch.float64)

    def is_bounded(self, manner="both"):
        below = torch.all(self.bounded_below)
        above = torch.all(self.bounded_above)
        if manner == "both":
            return below and above
        elif manner == "below":
            return below
        elif manner == "above":
            return above
        else:
            raise ValueError("manner is not in {"below", "above", "both"}")

    def sample(self):
        """
        Generates a single random sample inside of the Box.
        In creating a sample of the box, each coordinate is sampled according to the
        form of the interval:

        * [a, b]: uniform distribution
        * [a, oo): shifted exponential distribution
        *(-00, b]: shifted negative exponential distribution
        * (-oo, oo): normal distribution
        """
        high = self.high.type(torch.float64) if self.dtype.is_floating_point else self.high.type(torch.int64) + 1
        sample = torch.empty(self.shape, dtype=torch.float64)

        # Masking arrays which classify the coordinates according to interval type
        unbounded = ~self.bounded_below & ~self.bounded_above
        upp_bounded = ~self.bounded_below & self.bounded_above
        low_bounded = self.bounded_below & ~self.bounded_above
        bounded = self.bounded_below & self.bounded_above

        # Vectorised sampling by interval type
        sample[unbounded] = torch.randn(unbounded[unbounded].shape, dtype=torch.float64)

        sample[low_bounded] = (-torch.rand(low_bounded[low_bounded].shape, dtype=torch.float64)).exponential_() + \
                              self.low[low_bounded]

        sample[upp_bounded] = self.high[upp_bounded] - (
            -torch.rand(upp_bounded[upp_bounded].shape, dtype=torch.float64)).exponential_()

        sample[bounded] = (self.low[bounded] - high[bounded]) * torch.rand(bounded[bounded].shape,
                                                                           dtype=torch.float64) + high[bounded]

        if not self.dtype.is_floting_point:  # integer
            sample = torch.floor(sample)

        return sample.type(self.dtype)

    def mutate(self, x, mask=None):
        if(mask==None):
          mask=torch.ones(x.shape).to(x.device)

        mutate_mask = mask*((torch.rand(self.shape) < self.indpb).type(torch.float64)).to(x.device)
        noise = torch.normal(self.mutation_mean, self.mutation_std).to(x.device)
        x = x.type(torch.float64) + mutate_mask * noise
        if not self.dtype.is_floating_point:  # integer
            x = torch.floor(x)
        x = x.type(self.dtype)
        if not self.contains(x):
            return self.clamp(x)
        else:
            return x

    def contains(self, x):
        if isinstance(x, list):
            x = torch.tensor(x)  # Promote list to array for contains check
        return x.shape == self.shape and torch.all(x >= torch.as_tensor(self.low, dtype=self.dtype, device=x.device)) and torch.all(x <= torch.as_tensor(self.high, dtype=self.dtype, device=x.device))

    def clamp(self, x):
        if self.is_bounded(manner="below"):
            x = torch.max(x, torch.as_tensor(self.low, dtype=self.dtype, device=x.device))
        if self.is_bounded(manner="above"):
            x = torch.min(x, torch.as_tensor(self.high, dtype=self.dtype, device=x.device))
        return x

    def __repr__(self):
        return "BoxSpace({}, {}, {}, {})".format(self.low.min(), self.high.max(), self.shape, self.dtype)

    def __eq__(self, other):
        return isinstance(other, BoxSpace) and (self.shape == other.shape) and torch.allclose(self.low,
                                                                                              other.low) and torch.allclose(
            self.high, other.high)


class DictSpace(Space):
    """
    A Dict dictionary of simpler spaces.

    Example usage:
    self.genome_space = spaces.DictSpace({"position": spaces.Discrete(2), "velocity": spaces.Discrete(3)})

    Example usage [nested]:
    self.nested_genome_space = spaces.DictSpace({
        'sensors':  spaces.DictSpace({
            'position': spaces.Box(low=-100, high=100, shape=(3,)),
            'velocity': spaces.Box(low=-1, high=1, shape=(3,)),
            'front_cam': spaces.Tuple((
                spaces.Box(low=0, high=1, shape=(10, 10, 3)),
                spaces.Box(low=0, high=1, shape=(10, 10, 3))
            )),
            'rear_cam': spaces.Box(low=0, high=1, shape=(10, 10, 3)),
        }),
        'ext_controller': spaces.MultiDiscrete((5, 2, 2)),
        'inner_state':spaces.DictSpace({
            'charge': spaces.Discrete(100),
            'system_checks': spaces.MultiBinary(10),
            'job_status': spaces.DictSpace({
                'task': spaces.Discrete(5),
                'progress': spaces.Box(low=0, high=100, shape=()),
            })
        })
    })
    """

    def __init__(self, spaces=None, **spaces_kwargs):
        assert (spaces is None) or (
            not spaces_kwargs), 'Use either DictSpace(spaces=dict(...)) or DictSpace(foo=x, bar=z)'
        if spaces is None:
            spaces = spaces_kwargs
        if isinstance(spaces, list):
            spaces = Dict(spaces)
        self.spaces = spaces
        for space in spaces.values():
            assert isinstance(space, Space), 'Values of the attrdict should be instances of gym.Space'
        Space.__init__(self, None, None)  # None for shape and dtype, since it'll require special handling

    def sample(self):
        return Dict([(k, space.sample()) for k, space in self.spaces.items()])

    def mutate(self, x):
        return Dict([(k, space.mutate(x[k])) for k, space in self.spaces.items()])

    def contains(self, x):
        if not isinstance(x, dict) or len(x) != len(self.spaces):
            return False
        for k, space in self.spaces.items():
            if k not in x:
                return False
            if not space.contains(x[k]):
                return False
        return True

    def clamp(self, x):
        return Dict([(k, space.clamp(x[k])) for k, space in self.spaces.items()])

    def __getitem__(self, key):
        return self.spaces[key]

    def __iter__(self):
        for key in self.spaces:
            yield key

    def __repr__(self):
        return "DictSpace(" + ", ".join([str(k) + ":" + str(s) for k, s in self.spaces.items()]) + ")"

    def __eq__(self, other):
        return isinstance(other, DictSpace) and self.spaces == other.spaces


class MultiDiscreteSpace(Space):
    """
    - The multi-discrete space consists of a series of discrete spaces with different number of possible instances in eachs
    - Can be initialized as

        MultiDiscreteSpace([ 5, 2, 2 ])

    """

    def __init__(self, nvec, mutation_mean=0.0, mutation_std=1.0, indpb=1.0):

        """
        nvec: vector of counts of each categorical variable
        """
        assert (torch.tensor(nvec) > 0).all(), 'nvec (counts) have to be positive'
        self.nvec = torch.as_tensor(nvec, dtype=torch.int64)
        self.mutation_std = mutation_std

        # mutation_mean: mean for the gaussian addition mutation
        # mutation_std: std for the gaussian addition mutation
        # indpb – independent probability for each attribute to be mutated.
        if isinstance(mutation_mean, numbers.Number):
            mutation_mean = torch.full(self.nvec.shape, mutation_mean, dtype=torch.float64)
        self.mutation_mean = torch.as_tensor(mutation_mean, dtype=torch.float64)
        if isinstance(mutation_std, numbers.Number):
            mutation_std = torch.full(self.nvec.shape, mutation_std, dtype=torch.float64)
        self.mutation_std = torch.as_tensor(mutation_std, dtype=torch.float64)
        if isinstance(indpb, numbers.Number):
            indpb = torch.full(self.nvec.shape, indpb, dtype=torch.float64)
        self.indpb = torch.as_tensor(indpb, dtype=torch.float64)

        super(MultiDiscreteSpace, self).__init__(self.nvec.shape, torch.int64)

    def sample(self):
        return (torch.rand(self.nvec.shape) * self.nvec).type(self.dtype)

    def mutate(self, x):
        mutate_mask = (torch.rand(self.shape) < self.indpb).to(x.device)
        noise = torch.normal(self.mutation_mean, self.mutation_std).to(x.device)
        x = x.type(torch.float64) + mutate_mask * noise
        x = torch.floor(x).type(self.dtype)
        if not self.contains(x):
            return self.clamp(x)
        else:
            return x

    def contains(self, x):
        if isinstance(x, list):
            x = torch.tensor(x)  # Promote list to array for contains check
        # if nvec is uint32 and space dtype is uint32, then 0 <= x < self.nvec guarantees that x
        # is within correct bounds for space dtype (even though x does not have to be unsigned)
        return x.shape == self.shape and (0 <= x).all() and (x < self.nvec).all()

    def clamp(self, x):
        x = torch.max(x, torch.as_tensor(0, dtype=self.dtype, device=x.device))
        x = torch.min(x, torch.as_tensor(self.nvec - 1, dtype=self.dtype, device=x.device))
        return x

    def __repr__(self):
        return "MultiDiscreteSpace({})".format(self.nvec)

    def __eq__(self, other):
        return isinstance(other, MultiDiscreteSpace) and torch.all(self.nvec == other.nvec)


class BoxGoalSpace(BoxSpace):
    def __init__(self, representation, autoexpand=True, low=0., high=0., shape=None, dtype=torch.float32):
        self.representation = representation
        self.autoexpand = autoexpand
        if shape is not None:
            if isinstance(shape, list) or isinstance(shape, tuple):
                assert len(shape) == 1 and shape[0] == self.representation.n_latents
            elif isinstance(shape, numbers.Number):
                assert shape == self.representation.n_latents
        BoxSpace.__init__(self, low=low, high=high, shape=(self.representation.n_latents,), dtype=dtype)

    def map(self, observations, **kwargs):
        embedding = self.representation.calc(observations, **kwargs)
        if self.autoexpand:
            embedding_c = embedding.detach()
            is_nan_mask = torch.isnan(embedding_c)
            if is_nan_mask.sum() > 0:
                embedding_c[is_nan_mask] = self.low[is_nan_mask]
                self.low = torch.min(self.low, embedding_c)
                embedding_c[is_nan_mask] = self.high[is_nan_mask]
                self.high = torch.max(self.high, embedding_c)
            else:
                self.low = torch.min(self.low, embedding_c)
                self.high = torch.max(self.high, embedding_c)
        return embedding

    def calc_distance(self, embedding_a, embedding_b, **kwargs):
        return self.representation.calc_distance(embedding_a, embedding_b, **kwargs)

    def sample(self):
        return BoxSpace.sample(self)



class RunDataEntry(Dict):
    """
    Class that specify for RunData entry in the DB
    """

    def __init__(self, db, id, policy_parameters, observations, **kwargs):
        """
        :param kwargs: flexible structure of the entry which might contain additional columns (eg: source_policy_idx, target_goal, etc.)
        """
        super().__init__(**kwargs)
        self.db = db
        self.id = id
        self.policy_parameters = policy_parameters
        self.observations = observations

class ExplorationDB:
    """
    Base of all Database classes.
    """

    @staticmethod
    def default_config():

        default_config = Dict()
        default_config.db_directory = "database"
        default_config.save_observations = True
        default_config.keep_saved_runs_in_memory = True
        default_config.memory_size_run_data = 'infinity'  # number of runs that are kept in memory: 'infinity' - no imposed limit, int - number of runs saved in memory
        default_config.load_observations = True  # if set to false observations are not loaded in the load() function

        return default_config

    def __init__(self, config={}, **kwargs):

        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        if self.config.memory_size_run_data != 'infinity':
            assert isinstance(self.config.memory_size_run_data,
                              int) and self.config.memory_size_run_data > 0, "config.memory_size_run_data must be set to infinity or to an integer >= 1"

        self.reset_empty_db()

    def reset_empty_db(self):
        self.runs = OrderedDict()
        self.run_ids = set()  # list with run_ids that exist in the db
        self.run_data_ids_in_memory = []  # list of run_ids that are hold in memory

    def add_run_data(self, id, policy_parameters, observations, **kwargs):

        run_data_entry = RunDataEntry(db=self, id=id, policy_parameters=policy_parameters, observations=observations,
                                      **kwargs)
        if id not in self.run_ids:
            self.add_run_data_to_memory(id, run_data_entry)
            self.run_ids.add(id)

        else:
            warnings.warn(f'/!\ id {id} already in the database: overwriting it with new run data !!!')
            self.add_run_data_to_memory(id, run_data_entry, replace_existing=True)

        self.save([id])  # TODO: modify if we do not want to automatically save after each run

    def add_run_data_to_memory(self, id, run_data, replace_existing=False):
        self.runs[id] = run_data
        if not replace_existing:
            self.run_data_ids_in_memory.insert(0, id)

        # remove last item from memory when not enough size
        if self.config.memory_size_run_data != 'infinity' and len(
                self.run_data_ids_in_memory) > self.config.memory_size_run_data:
            del (self.runs[self.run_data_ids_in_memory[-1]])
            del (self.run_data_ids_in_memory[-1])

    def save(self, run_ids=None):
        # the run data entry is save in 2 files: 'run_*_data*' (general data dict such as run parameters -> for now json) and ''run_*_observations*' (observation data dict -> for now npz)
        if run_ids is None:
            run_ids = []

        for run_id in run_ids:
            self.save_run_data_to_db(run_id)
            if self.config.save_observations:
                self.save_observations_to_db(run_id)

        if not self.config.keep_saved_runs_in_memory:
            for run_id in run_ids:
                del self.runs[run_id]
            self.run_data_ids_in_memory = []

    def save_run_data_to_db(self, run_id):
        run_data = self.runs[run_id]

        # add all data besides the observations
        save_dict = dict()
        for data_name, data_value in run_data.items():
            if data_name not in ['observations', 'db']:
                save_dict[data_name] = data_value
        filename = 'run_{:07d}_data.pickle'.format(run_id)
        filepath = os.path.join(self.config.db_directory, filename)

        torch.save(save_dict, filepath)

    def save_observations_to_db(self, run_id):
        # TODO: create an abstract observation class with a save method for observations that are not numpy array
        run_data = self.runs[run_id]

        filename = 'run_{:07d}_observations.pickle'.format(run_id)
        filepath = os.path.join(self.config.db_directory, filename)

        torch.save(run_data.observations, filepath)

    def load(self, run_ids=None, map_location="cpu"):
        """
        Loads the data base.
        :param run_ids:  IDs of runs for which the data should be loaded into the memory.
                        If None is given, all ids are loaded (up to the allowed memory size).
        :param map_location: device on which the database is loaded
        """

        if run_ids is not None:
            assert isinstance(run_ids, list), "run_ids must be None or a list"

        # set run_ids from the db directory and empty memory
        self.run_ids = self.load_run_ids_from_db()
        self.runs = OrderedDict()
        self.run_data_ids_in_memory = []

        if run_ids is None:
            run_ids = self.run_ids

        if len(run_ids) > 0:

            if self.config.memory_size_run_data != 'infinity' and len(run_ids) > self.config.memory_size_run_data:
                # only load the maximum number of run_data into the memory
                run_ids = list(run_ids)[-self.config.memory_size_run_data:]

            self.load_run_data_from_db(run_ids, map_location=map_location)

    def load_run_ids_from_db(self):
        run_ids = set()

        file_matches = glob(os.path.join(self.config.db_directory, 'run_*_data*'))
        for match in file_matches:
            id_as_str = re.findall('_(\d+).', match)
            if len(id_as_str) > 0:
                run_ids.add(int(id_as_str[
                                    -1]))  # use the last find, because ther could be more number in the filepath, such as in a directory name

        return run_ids

    def load_run_data_from_db(self, run_ids, map_location="cpu"):
        """Loads the data for a list of runs and adds them to the memory."""

        if not os.path.exists(self.config.db_directory):
            raise Exception('The directory {!r} does not exits! Cannot load data.'.format(self.config.db_directory))

        print('Loading Data: ')
        for run_id in tqdm(run_ids):
            # load general data (run parameters and others)
            filename = 'run_{:07d}_data.pickle'.format(run_id)
            filepath = os.path.join(self.config.db_directory, filename)

            if os.path.exists(filepath):
                run_data_kwargs = torch.load(filepath, map_location=map_location)
            else:
                run_data_kwargs = {'id': None, 'policy_parameters': None}

            if self.config.load_observations:
                filename_obs = 'run_{:07d}_observations.pickle'.format(run_id)
                filepath_obs = os.path.join(self.config.db_directory, filename_obs)

                # load observations
                if os.path.exists(filepath_obs):
                    observations = torch.load(filepath_obs, map_location=map_location)
                else:
                    observations = None
            else:
                observations = None

            # create run data and add it to memory
            run_data = RunDataEntry(self, observations=observations, **run_data_kwargs)
            self.add_run_data_to_memory(run_id, run_data)

            if not self.config.keep_saved_runs_in_memory:
                del self.runs[run_id]
                del self.run_data_ids_in_memory[0]

        return




### DEMO COD:

import matplotlib.cm as cm
import cv2

def main(SX, SY, mode, boarders, list_kernels=range(10), creaFile="crea1.pickle",
         mode = "none", zoom =1):
    lenia_config = Lenia_C.default_config()
    lenia_config.SX = SX
    lenia_config.SY = SY
    lenia_config.final_step = 200
    lenia_config.version = "pytorch_fft"
    lenia_config.nb_kernels = len(list_kernels)
    initialization_space_config = Dict()
    initialization_space = LeniaInitializationSpace(config=initialization_space_config)
    system = Lenia_C(initialization_space, config=lenia_config, device = "cuda")
    a = torch.load(creaFile)

    # b=torch.load("run_0000179_data.pickle")
    policy_parameters = Dict.fromkeys(['initialization', 'update_rule'])
    policy_parameters['initialization']=a['policy_parameters']['initialization']
    policy_parameters['update_rule']=a['policy_parameters']['update_rule']

    # random_kernels=torch.randperm(10)[:9]

    policy_parameters['update_rule']['R']=(policy_parameters['update_rule']['R']+15)*zoom-15
    init_s=policy_parameters['initialization'].init.cpu().numpy()*1.0


    width = int(init_s.shape[1]*zoom)
    height = int(init_s.shape[0]* zoom)
    dim = (width, height)
    # resize image
    resized = cv2.resize(init_s,dim)
    init_f=torch.tensor(resized).to('cuda')

    for k in policy_parameters["update_rule"].keys():
        if(k!="R" and k != "T"):
            policy_parameters["update_rule"][k] = policy_parameters["update_rule"][k][list_kernels]
        policy_parameters["update_rule"][k] = policy_parameters["update_rule"][k].to("cuda")

    system.reset(initialization_parameters=policy_parameters["initialization"],
                 update_rule_parameters = policy_parameters["update_rule"])
    creature_x = -40
    creature_y = -40
    data_split=["a", "a"]


  while True:
    if(mode=='draw'):
      print('you can draw on the canvas or click on circle to go to circle mode')
      print('click on video once you re done')
      data=['2']
      while(data[0]=='2'):

        cv_HTML=canvas_html % (SY, SX,system.config.final_step)

        for i in range(10):
          cv_HTML=cv_HTML+kernels_HTML.format(i=i,
                                      h=policy_parameters['update_rule']['h'][i],
                                      m=policy_parameters['update_rule']['m'][i],
                                      s=policy_parameters['update_rule']['s'][i],
                                      r=policy_parameters['update_rule']['r'][i],
                                      rk1=policy_parameters['update_rule']['rk'][i][0],
                                      rk2=policy_parameters['update_rule']['rk'][i][1],
                                      rk3=policy_parameters['update_rule']['rk'][i][2],
                                      w1=policy_parameters['update_rule']['w'][i][0],
                                      w2=policy_parameters['update_rule']['w'][i][1],
                                      w3=policy_parameters['update_rule']['w'][i][2],
                                      b1=policy_parameters['update_rule']['b'][i][0],
                                      b2=policy_parameters['update_rule']['b'][i][1],
                                      b3=policy_parameters['update_rule']['b'][i][2])

        cv_HTML=cv_HTML+end_HTML %( 8,data_split[-2]+","+data_split[-1])
        # print(cv_HTML)

        html_object=HTML(cv_HTML)
        # print(canvas_html % (SY, SX, 8,creature_x,creature_y,data_url))
        display(html_object)


        data = eval_js('data')
        if(data[0]=='2'):
          a=torch.load(creaFile)
          policy_parameters['update_rule']=a['policy_parameters']['update_rule']
          for k in policy_parameters['update_rule'].keys():


            if(k!='R' and k!='T'):

              policy_parameters['update_rule'][k]=policy_parameters['update_rule'][k][list_kernels]
            policy_parameters['update_rule'][k]=policy_parameters['update_rule'][k].to('cuda')
          clear_output(wait=False)
          data_url=data[6:]
          data_split=data_url.split(',')
          system.config.final_step=int(data[1:5])


      if(data[0]=='0'):
        break
      else:
        data_url=data[6:]
        data_split=data_url.split(',')
        system.config.final_step=int(data[1:5])
        for i in range(10):
          policy_parameters['update_rule']['h'][i]=float(data_split[i*13])
          policy_parameters['update_rule']['m'][i]=float(data_split[i*13+1])
          policy_parameters['update_rule']['s'][i]=float(data_split[i*13+2])
          policy_parameters['update_rule']['r'][i]=float(data_split[i*13+3])
          policy_parameters['update_rule']['rk'][i][0]=float(data_split[i*13+4])
          policy_parameters['update_rule']['rk'][i][1]=float(data_split[i*13+5])
          policy_parameters['update_rule']['rk'][i][2]=float(data_split[i*13+6])
          policy_parameters['update_rule']['w'][i][0]=float(data_split[i*13+7])
          policy_parameters['update_rule']['w'][i][1]=float(data_split[i*13+8])
          policy_parameters['update_rule']['w'][i][2]=float(data_split[i*13+9])
          policy_parameters['update_rule']['b'][i][0]=float(data_split[i*13+10])
          policy_parameters['update_rule']['b'][i][1]=float(data_split[i*13+11])
          policy_parameters['update_rule']['b'][i][2]=float(data_split[i*13+12])

      system.reset(initialization_parameters=policy_parameters['initialization'],update_rule_parameters=policy_parameters['update_rule'])
      # print(data_url)

      binary = b64decode(data_split[-1])
      with open("laby.png", 'wb') as f:
        f.write(binary)
      img = Image.open('laby.png')
      img=np.array(img)

      # plt.imshow(img[:,:,:3])
      # plt.show()
      # plt.imshow(img[:,:,-1])
      # plt.show()
      # if(img[0,0,1]>100):
      #   break

      if(np.all(img[:,:,0]<240)):
        print("you didn't put the creature, creature put automatically in the bottom right corner")
      else:
        system.init_loca=[]
        for i in range(1,SX-40):
          for j in range(1,SY-40):
              if(img[i,j,0]>240 and img[i-1,j,0]<240 and img[i,j-1,0]<240 and img[i+1,j,0]>240 and img[i,j+1,0]>240):
                system.init_loca.append((i,j))

      img=((img[:,:,-1]>0).astype(np.float)-(img[:,:,0]>240).astype(np.float))

      system.init_wall=torch.tensor(img)

    if(mode=='random'):
      nb_obstacle=int(input("number of obstacles (100 is interesting) "))
      radius_obstacle=int(input("radius of obstacles (10 is good)  "))
      system.random_obstacle(nb_obstacle,radius_obstacle)

    if(borders):
      system.init_wall[:,:4]=1
      system.init_wall[:,-4:]=1
      system.init_wall[-4:,:]=1
      system.init_wall[:4,:]=1
    print('Lenia running')
    time_b=time.time()
    with torch.no_grad():
      system.generate_init_state()
      system.state[0,:,:,0]=0
      print(system.init_loca)
      for loca in system.init_loca:
          system.state[0,loca[0]:loca[0]+init_f.shape[0],loca[1]:loca[1]+init_f.shape[1],0]=init_f
      observations = system.run()

    print('Creating video')

    time_lenia=time.time()-time_b

    cmap = cm.get_cmap('jet')
    with VideoWriter("out.mp4", 30.0) as vid:
      for timestep in range(observations["states"].shape[0]):

        # rgb_im = im_from_array_with_colormap(a["states"][timestep,:,:,0].detach().cpu().numpy(), colormap)
        # rgb_im = np.array(rgb_im.convert("RGB"))
        # rgb_arr = np.array(rgb_im.convert("RGB"))
        # print(a["states"][timestep,:,:,0].detach().cpu().unsqueeze(-1).numpy().repeat(2,2).shape)



        rgb_im=np.concatenate([observations["states"][timestep,:,:,0].detach().cpu().unsqueeze(-1).numpy().repeat(2,2),observations["states"][timestep,:,:,1].detach().cpu().unsqueeze(-1).numpy()],axis=2)
        # rgb_im=cmap(observations["states"][timestep,:,:,0].detach().cpu().numpy())[:,:,:3]
        # rgb_im=np.clip(rgb_im-observations["states"][timestep,:,:,1].detach().cpu().unsqueeze(-1).numpy(),0,1)
        vid.add(rgb_im)
      clear_output(wait=False)
      print(policy_parameters['update_rule'])
      vid.show()

    cmap = cm.get_cmap('magma')
    # for i in range(10):
    #   with VideoWriter("out.mp4", 30.0) as vid:
    #     for timestep in range(observations["states"].shape[0]):
    #       # print(observations["kernels"][i].shape)
    #       # rgb_im=(observations["kernels"][timestep,i].detach().cpu().unsqueeze(-1).numpy().repeat(3,2))
    #       rgb_im=cmap(observations["kernels"][timestep,i].detach().cpu().numpy())


    #       # print(rgb_im.shape)
    #       vid.add(rgb_im[:,:,:3])
    #     vid.show()
    # with VideoWriter("out.mp4", 30.0) as vid:
    #     for timestep in range(observations["states"].shape[0]):
    #       # print(observations["kernels"][i].shape)
    #       rgb_im=(observations["growth"][timestep].detach().cpu().unsqueeze(-1).numpy().repeat(3,2))
    #       # rgb_im=cmap(observations["kernels"][timestep,i].detach().cpu().numpy())


    #       # print(rgb_im.shape)
    #       vid.add(rgb_im[:,:,:3])
    #     vid.show()

    if(modeb=='growth' or modeb=='both'):
      observations["kernels"][0,:,:,:]=-1
      observations["growth"][0,:,:]=0
      min=torch.min(observations["kernels"])
      max=torch.max(observations["kernels"])
      observations["kernels"]=(observations["kernels"]-min)/(max-min)
      colorbar=np.linspace(0,1,4*SX)
      colorbar=np.expand_dims(colorbar,-1)
      colorbar=colorbar.repeat(50,axis=-1)
      colorbar=cmap(colorbar)[:,:,:3]
      colorbar=cv2.putText(colorbar, #numpy array on which text is written
                str(round(max.item(),3)), #text
                (17,4*SX-10), #position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX, #font family
                0.3, #font size
                (0, 0,0, 255), #font color
                1)
      colorbar=cv2.putText(colorbar, #numpy array on which text is written
                str(0), #text
                (17,2*SX), #position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX, #font family
                0.3, #font size
                (209, 80, 250, 255), #font color
                1) #font stroke
      colorbar=cv2.putText(colorbar, #numpy array on which text is written
                str(round(min.item(),3)), #text
                (17,15), #position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX, #font family
                0.3, #font size
                (209, 80, 250, 255), #font color
                1).get().astype('f') #font stroke

      with VideoWriter("out.mp4", 30.0) as vid:
        for timestep in range(observations["states"].shape[0]):
          im=np.concatenate([observations["states"][timestep,:,:,0].detach().cpu().unsqueeze(-1).numpy().repeat(2,2),observations["states"][timestep,:,:,1].detach().cpu().unsqueeze(-1).numpy()],axis=2)
          kern=cmap(observations["kernels"][timestep,:,:,:].detach().cpu().numpy())[:,:,:,:3]
          growth=(observations["growth"][timestep].detach().cpu().unsqueeze(-1).numpy().repeat(3,2))
          rgb_im=np.zeros((4*SX,3*SY,3))
          for i in range(10):
            position = (10,50)
            kern[i]=cv2.putText(kern[i], #numpy array on which text is written
                "h= "+str(round(policy_parameters['update_rule']['h'][i].item(),3))+ " m = "+str(round(policy_parameters['update_rule']['m'][i].item(),3))+" s = "+str(round(policy_parameters['update_rule']['s'][i].item(),3) ), #text
                position, #position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX, #font family
                0.3, #font size
                (209, 80, 250, 255), #font color
                1).get().astype('f') #font stroke
            kern[i][img>0.5]=[0,0,1]
            growth[img>0.5]=[0,0,1]

          for i in range(3):
            rgb_im[:SX,i*SY:(i+1)*SY]=kern[i]
          rgb_im[SX:2*SX,:SY]=kern[3]
          rgb_im[SX:2*SX,SY:2*SY]=im
          rgb_im[SX:2*SX,2*SY:3*SY]=kern[4]
          rgb_im[2*SX:3*SX,:SY]=kern[5]
          rgb_im[2*SX:3*SX,SY:2*SY]=growth
          rgb_im[2*SX:3*SX,2*SY:3*SY]=kern[6]
          for i in range(3):
            rgb_im[3*SX:4*SX,i*SY:(i+1)*SY]=kern[7+i]
          rgb_im=np.concatenate([rgb_im,colorbar],axis=1)



          vid.add(rgb_im)
        vid.show()
    if(modeb=='sum' or modeb=='both'):
      observations["kernel_neighb"][0,:,:,:]=0
      observations["growth"][0,:,:]=0
      min=torch.min(observations["kernel_neighb"])
      max=torch.max(observations["kernel_neighb"])
      observations["kernel_neighb"]=(observations["kernel_neighb"]-min)/(max-min)
      colorbar=np.linspace(0,1,4*SX)
      colorbar=np.expand_dims(colorbar,-1)
      colorbar=colorbar.repeat(50,axis=-1)
      colorbar=cmap(colorbar)[:,:,:3]
      colorbar=cv2.putText(colorbar, #numpy array on which text is written
                str(round(max.item(),3)), #text
                (17,4*SX-10), #position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX, #font family
                0.3, #font size
                (0, 0,0, 255), #font color
                1)
      colorbar=cv2.putText(colorbar, #numpy array on which text is written
                str(round(min.item(),3)), #text
                (17,15), #position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX, #font family
                0.3, #font size
                (209, 80, 250, 255), #font color
                1).get().astype('f') #font stroke
      with VideoWriter("out.mp4", 30.0) as vid:
        for timestep in range(observations["states"].shape[0]):
          im=np.concatenate([observations["states"][timestep,:,:,0].detach().cpu().unsqueeze(-1).numpy().repeat(2,2),observations["states"][timestep,:,:,1].detach().cpu().unsqueeze(-1).numpy()],axis=2)
          kern=cmap(observations["kernel_neighb"][timestep,:,:,:].detach().cpu().numpy())[:,:,:,:3]
          growth=((policy_parameters['update_rule']['h'].unsqueeze(-1).unsqueeze(-1)*observations["kernel_neighb"][timestep,:,:,:]).sum(0).detach().cpu().unsqueeze(-1).numpy().repeat(3,2))
          rgb_im=np.zeros((4*SX,3*SY,3))
          for i in range(10):
            position = (10,50)
            kern[i][img>0.5]=[0,0,1]

            kern[i]=cv2.putText(kern[i], #numpy array on which text is written
                "h= "+str(round(policy_parameters['update_rule']['h'][i].item(),3))+ " m = "+str(round(policy_parameters['update_rule']['m'][i].item(),3))+" s = "+str(round(policy_parameters['update_rule']['s'][i].item(),3) ), #text
                position, #position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX, #font family
                0.3, #font size
                (209, 80, 250, 255), #font color
                1).get().astype('f') #font stroke

          for i in range(3):
            rgb_im[:SX,i*SY:(i+1)*SY]=kern[i]
          rgb_im[SX:2*SX,:SY]=kern[3]
          rgb_im[SX:2*SX,SY:2*SY]=im
          rgb_im[SX:2*SX,2*SY:3*SY]=kern[4]
          rgb_im[2*SX:3*SX,:SY]=kern[5]
          rgb_im[2*SX:3*SX,SY:2*SY]=growth
          rgb_im[2*SX:3*SX,2*SY:3*SY]=kern[6]
          for i in range(3):
            rgb_im[3*SX:4*SX,i*SY:(i+1)*SY]=kern[7+i]

          rgb_im=np.concatenate([rgb_im,colorbar],axis=1)



          vid.add(rgb_im)
        vid.show()
    print(data)
    print('computation of lenia took '+str(time_lenia))
    if(mode=='random'):
      break

class LeniaInitializationSpace(DictSpace):
    """ Class for initialization space that allows to sample and clip the initialization"""
    @staticmethod
    def default_config():
        default_config = Dict()
        default_config.neat_config = None
        default_config.cppn_n_passes = 2
        return default_config

    def __init__(self,init_size=40,  config={}, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        spaces = Dict(
            # cppn_genome = LeniaCPPNInitSpace(self.config)
            init=BoxSpace(low=0.0,high=1.0,shape=(init_size,init_size),mutation_mean=torch.zeros((40,40)),mutation_std=torch.ones((40,40))*0.01,indpb=0.0,dtype=torch.float32)
        )

        DictSpace.__init__(self, spaces=spaces)



""" =============================================================================================
Lenia Update Rule Space: 
============================================================================================= """


class LeniaUpdateRuleSpace(DictSpace):
    """ Space associated to the parameters of the update rule"""
    @staticmethod
    def default_config():
        default_config = Dict()
        return default_config

    def __init__(self,nb_k=10, config={}, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)

        spaces = Dict(
            R = DiscreteSpace(n=25, mutation_mean=0.0, mutation_std=0.01, indpb=0.01),
            c0= MultiDiscreteSpace(nvec=[1]*nb_k, mutation_mean=torch.zeros((nb_k,)), mutation_std=0.1*torch.ones((nb_k,)), indpb=0.1),
            c1= MultiDiscreteSpace(nvec=[1]*nb_k, mutation_mean=torch.zeros((nb_k,)), mutation_std=0.1*torch.ones((nb_k,)), indpb=0.1),
            T = BoxSpace(low=1.0, high=10.0, shape=(), mutation_mean=0.0, mutation_std=0.1, indpb=0.01, dtype=torch.float32),
            rk = BoxSpace(low=0, high=1, shape=(nb_k,3), mutation_mean=torch.zeros((nb_k,3)), mutation_std=0.2*torch.ones((nb_k,3)), indpb=1, dtype=torch.float32),
            b = BoxSpace(low=0.0, high=1.0, shape=(nb_k,3), mutation_mean=torch.zeros((nb_k,3)), mutation_std=0.2*torch.ones((nb_k,3)), indpb=1, dtype=torch.float32),
            w = BoxSpace(low=0.01, high=0.5, shape=(nb_k,3), mutation_mean=torch.zeros((nb_k,3)), mutation_std=0.2*torch.ones((nb_k,3)), indpb=1, dtype=torch.float32),
            m = BoxSpace(low=0.05, high=0.5, shape=(nb_k,), mutation_mean=torch.zeros((nb_k,)), mutation_std=0.2*torch.ones((nb_k,)), indpb=1, dtype=torch.float32),
            s = BoxSpace(low=0.001, high=0.18, shape=(nb_k,), mutation_mean=torch.zeros((nb_k,)), mutation_std=0.01**torch.ones((nb_k,)), indpb=0.1, dtype=torch.float32),
            h = BoxSpace(low=0, high=1.0, shape=(nb_k,), mutation_mean=torch.zeros((nb_k,)), mutation_std=0.2*torch.ones((nb_k,)), indpb=0.1, dtype=torch.float32),
            r = BoxSpace(low=0.2, high=1.0, shape=(nb_k,), mutation_mean=torch.zeros((nb_k,)), mutation_std=0.2*torch.ones((nb_k,)), indpb=1, dtype=torch.float32)
            #kn = DiscreteSpace(n=4, mutation_mean=0.0, mutation_std=0.1, indpb=1.0),
            #gn = DiscreteSpace(n=3, mutation_mean=0.0, mutation_std=0.1, indpb=1.0),
        )

        DictSpace.__init__(self, spaces=spaces)
    def mutate(self,x):
      mask=(x['s']>0.04).float()*(torch.rand(x['s'].shape[0])<0.25).float().to(x['s'].device)
      param=[]
      for k, space in self.spaces.items():
        if(k=="R" or k=="c0" or k=="c1" or k=="T"):
          param.append((k, space.mutate(x[k])))
        elif(k=='rk' or k=='w' or k=='b'):
          param.append((k, space.mutate(x[k],mask.unsqueeze(-1))))
        else:
          param.append((k, space.mutate(x[k],mask)))

      return Dict(param)


""" =============================================================================================
Lenia Main
============================================================================================= """

bell = lambda x, m, s: torch.exp(-((x-m)/s)**2 / 2)
# Lenia family of functions for the kernel K and for the growth mapping g
kernel_core = {
    0: lambda u: (4 * u * (1 - u)) ** 4,  # polynomial (quad4)
    1: lambda u: torch.exp(4 - 1 / (u * (1 - u))),  # exponential / gaussian bump (bump4)
    2: lambda u, q=1 / 4: (u >= q).float() * (u <= 1 - q).float(),  # step (stpz1/4)
    3: lambda u, q=1 / 4: (u >= q).float() * (u <= 1 - q).float() + (u < q).float() * 0.5,  # staircase (life)
    4: lambda u: torch.exp(-(u-0.5)**2/0.2),
    8: lambda u: (torch.sin(10*u)+1)/2,
    9: lambda u: (a*torch.sin((u.unsqueeze(-1)*5*b+c)*np.pi)).sum(-1)/(2*a.sum())+1/2

}
field_func = {
    0: lambda n, m, s: torch.max(torch.zeros_like(n), 1 - (n - m) ** 2 / (9 * s ** 2)) ** 4 * 2 - 1, # polynomial (quad4)
    1: lambda n, m, s: torch.exp(- (n - m) ** 2 / (2 * s ** 2)-1e-3) * 2 - 1,  # exponential / gaussian (gaus)
    2: lambda n, m, s: (torch.abs(n - m) <= s).float() * 2 - 1 , # step (stpz)
    3: lambda n, m, s: - torch.clamp(n-m,0,1)*s #food eating kernl
}

# ker_c =lambda r,a,b,c :(a*torch.sin((r.unsqueeze(-1)*5*b+c)*np.pi)).sum(-1)/(2*a.sum())+1/2
ker_c= lambda x,r,w,b : (b*torch.exp(-((x.unsqueeze(-1)-r)/w)**2 / 2)).sum(-1)

class Dummy_init_mod(torch.nn.Module):
  def __init__(self,init):
    torch.nn.Module.__init__(self)
    self.register_parameter('init', torch.nn.Parameter(init))


# Lenia Step FFT version (faster)
class LeniaStepFFTC(torch.nn.Module):
    """ Module pytorch that computes one Lenia Step with the fft version"""

    def __init__(self,C, R, T,c0,c1,r,rk, b,w,h, m, s, gn, is_soft_clip=False, SX=256, SY=256, device='cpu'):
        torch.nn.Module.__init__(self)

        self.register_buffer('R', R)
        self.register_buffer('T', T)
        self.register_buffer('c0', c0)
        self.register_buffer('c1', c1)
        # self.register_buffer('r', r)
        self.register_parameter('r', torch.nn.Parameter(r))
        self.register_parameter('rk', torch.nn.Parameter(rk))
        self.register_parameter('b', torch.nn.Parameter(b))
        self.register_parameter('w', torch.nn.Parameter(w))
        self.register_parameter('h', torch.nn.Parameter(h))
        self.register_parameter('m', torch.nn.Parameter(m))
        self.register_parameter('s', torch.nn.Parameter(s))

        self.gn = 1
        self.nb_k=c0.shape[0]

        self.SX = SX
        self.SY = SY

        self.is_soft_clip = is_soft_clip
        self.C=C

        self.device = device
        self.to(device)
        self.kernels=torch.zeros((self.nb_k,self.SX,self.SY,2)).to(self.device)

        self.compute_kernel()
        self.compute_kernel_env()

    def compute_kernel_env(self):
      """ computes the kernel and the kernel FFT of the environnement from the parameters"""
      x = torch.arange(self.SX).to(self.device)
      y = torch.arange(self.SY).to(self.device)
      xx = x.view(-1, 1).repeat(1, self.SY)
      yy = y.repeat(self.SX, 1)
      X = (xx - int(self.SX / 2)).float()
      Y = (yy - int(self.SY / 2)).float()
      D = torch.sqrt(X ** 2 + Y ** 2)/(4)
      kernel = torch.sigmoid(-(D-1)*10) * ker_c(D,torch.tensor(np.array([0,0,0])).to(self.device),torch.tensor(np.array([0.5,0.1,0.1])).to(self.device),torch.tensor(np.array([1,0,0])).to(self.device))
      kernel_sum = torch.sum(kernel)
      kernel_norm = (kernel / kernel_sum).unsqueeze(0)
      kernel_FFT = torch.rfft(kernel_norm, signal_ndim=2, onesided=False).to(self.device)
      self.kernel_wall=kernel_FFT




    def compute_kernel(self):
      """ computes the kernel and the kernel FFT of the learnable channels from the parameters"""
      x = torch.arange(self.SX).to(self.device)
      y = torch.arange(self.SY).to(self.device)
      xx = x.view(-1, 1).repeat(1, self.SY)
      yy = y.repeat(self.SX, 1)
      X = (xx - int(self.SX / 2)).float()
      Y = (yy - int(self.SY / 2)).float()
      self.kernels=torch.zeros((self.nb_k,self.SX,self.SY,2)).to(self.device)


      for i in range(self.nb_k):
        # distance to center in normalized space
        D = torch.sqrt(X ** 2 + Y ** 2)/ ((self.R+15)*self.r[i])

        kernel = torch.sigmoid(-(D-1)*10) * ker_c(D,self.rk[i],self.w[i],self.b[i])
        kernel_sum = torch.sum(kernel)


        # normalization of the kernel
        kernel_norm = (kernel / kernel_sum).unsqueeze(0).unsqueeze(0)
        # plt.imshow(kernel_norm[0,0].detach().cpu()*100)
        # plt.show()


        # fft of the kernel
        kernel_FFT = torch.rfft(kernel_norm, signal_ndim=2, onesided=False).to(self.device)
        self.kernels[i]=kernel_FFT



    def forward(self, input):

        self.D=torch.zeros(input.shape).to(self.device)
        self.Dn=torch.zeros(self.C)

        world_FFT = [torch.rfft(input[:,:,:,i], signal_ndim=2, onesided=False) for i in range(self.C)]





        ## speed up of the update for 1 channel creature by multiplying by all the kernel FFT

        #channel 0 is the learnable channel
        world_FFT_c = world_FFT[0]
        #multiply the FFT of the world and the kernels
        potential_FFT = complex_mult_torch(self.kernels, world_FFT_c)
        #ifft + realign
        potential = torch.irfft(potential_FFT, signal_ndim=2, onesided=False)
        potential = roll_n(potential, 2, potential.size(2) // 2)
        potential = roll_n(potential, 1, potential.size(1) // 2)
        #growth function
        gfunc = field_func[min(self.gn, 3)]
        field = gfunc(potential, self.m.unsqueeze(-1).unsqueeze(-1), self.s.unsqueeze(-1).unsqueeze(-1))
        #add the growth multiplied by the weight of the rule to the total growth
        self.D[:,:,:,0]=(self.h.unsqueeze(-1).unsqueeze(-1)*field).sum(0,keepdim=True)
        self.Dn[0]=self.h.sum()






        ###Base version for the case where we want the learnable creature to be  multi channel (which is not used in this notebook)

        # for i in range(self.nb_k):
        #   c0b=int((self.c0[i]))
        #   c1b=int((self.c1[i]))

        #   world_FFT_c = world_FFT[c0b]
        #   potential_FFT = complex_mult_torch(self.kernels[i].unsqueeze(0), world_FFT_c)

        #   potential = torch.irfft(potential_FFT, signal_ndim=2, onesided=False)
        #   potential = roll_n(potential, 2, potential.size(2) // 2)
        #   potential = roll_n(potential, 1, potential.size(1) // 2)


        #   gfunc = field_func[min(self.gn, 3)]
        #   field = gfunc(potential, self.m[i], self.s[i])

        #   self.D[:,:,:,c1b]=self.D[:,:,:,c1b]+self.h[i]*field
        #   self.Dn[c1b]=self.Dn[c1b]+self.h[i]




        #apply wall
        world_FFT_c = world_FFT[self.C-1]
        potential_FFT = complex_mult_torch(self.kernel_wall, world_FFT_c)
        potential = torch.irfft(potential_FFT, signal_ndim=2, onesided=False)
        potential = roll_n(potential, 2, potential.size(2) // 2)
        potential = roll_n(potential, 1, potential.size(1) // 2)
        gfunc = field_func[3]
        field = gfunc(potential, 1e-8, 10)
        for i in range(self.C-1):
          c1b=i
          self.D[:,:,:,c1b]=self.D[:,:,:,c1b]+1*field
          self.Dn[c1b]=self.Dn[c1b]+1


        ## Add the total growth to the current state
        if not self.is_soft_clip:

            output_img = torch.clamp(input + (1.0 / self.T) * self.D, min=0., max=1.)
            # output_img = input + (1.0 / self.T) * ((self.D/self.Dn+1)/2-input)

        else:
            output_img = torch.sigmoid((input + (1.0 / self.T) * self.D-0.5)*10)
             # output_img = torch.tanh(input + (1.0 / self.T) * self.D)


        return output_img





class Lenia_C(torch.nn.Module):

    @staticmethod
    def default_config():
        default_config = Dict()
        default_config.version = 'pytorch_fft'  # "pytorch_fft", "pytorch_conv2d"
        default_config.SX = 256
        default_config.SY = 256
        default_config.final_step = 40
        default_config.C = 2
        return default_config


    def __init__(self, initialization_space=None, update_rule_space=None, nb_k=10,init_size=40, config={}, device=torch.device('cpu'), **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)
        torch.nn.Module.__init__(self)
        self.device = device
        self.init_size=init_size
        if initialization_space is not None:
            self.initialization_space = initialization_space
        else:
            self.initialization_space = LeniaInitializationSpace(self.init_size)

        if update_rule_space is not None:
            self.update_rule_space = update_rule_space
        else:
            self.update_rule_space = LeniaUpdateRuleSpace(nb_k)

        self.run_idx = 0
        self.init_wall=torch.zeros((self.config.SX,self.config.SY))
        #reset with no argument to sample random parameters
        self.reset()
        self.to(self.device)


    def reset(self, initialization_parameters=None, update_rule_parameters=None):
        # call the property setters
        if(initialization_parameters is not None):
          self.initialization_parameters = initialization_parameters
        else:
          self.initialization_parameters = self.initialization_space.sample()

        if(update_rule_parameters is not None):
          self.update_rule_parameters = update_rule_parameters
        else:
          policy_parameters = Dict.fromkeys(['update_rule'])
          policy_parameters['update_rule'] = self.update_rule_space.sample()
          #divide h by 3 at the beginning as some unbalanced kernels can easily kill
          policy_parameters['update_rule'].h =policy_parameters['update_rule'].h/3
          self.update_rule_parameters = policy_parameters['update_rule']

        # initialize Lenia CA with update rule parameters
        if self.config.version == "pytorch_fft":
            lenia_step = LeniaStepFFTC(self.config.C,self.update_rule_parameters['R'], self.update_rule_parameters['T'],self.update_rule_parameters['c0'],self.update_rule_parameters['c1'], self.update_rule_parameters['r'], self.update_rule_parameters['rk'], self.update_rule_parameters['b'], self.update_rule_parameters['w'],self.update_rule_parameters['h'], self.update_rule_parameters['m'],self.update_rule_parameters['s'],1, is_soft_clip=False, SX=self.config.SX, SY=self.config.SY, device=self.device)
        self.add_module('lenia_step', lenia_step)

        # initialize Lenia initial state with initialization_parameters
        init = self.initialization_parameters['init']
        # initialization_cppn = pytorchneat.rnn.RecurrentNetwork.create(cppn_genome, self.initialization_space.config.neat_config, device=self.device)
        self.add_module('initialization', Dummy_init_mod(init))

        # push the nn.Module and the available device
        self.to(self.device)
        self.generate_init_state()

    def  random_obstacle(self,nb_obstacle=6):
      self.init_wall=torch.zeros((self.config.SX,self.config.SY))

      x = torch.arange(self.config.SX)
      y = torch.arange(self.config.SY)
      xx = x.view(-1, 1).repeat(1, self.config.SY)
      yy = y.repeat(self.config.SX, 1)
      for i in range(nb_obstacle):
        X = (xx - int(torch.rand(1)*self.config.SX )).float()
        Y = (yy - int(torch.rand(1)*self.config.SY/2)).float()
        D = torch.sqrt(X ** 2 + Y ** 2)/10
        mask=(D<1).float()
        self.init_wall=torch.clamp(self.init_wall+mask,0,1)




    def generate_init_state(self,X=105,Y=180):
        init_state = torch.zeros( 1,self.config.SX, self.config.SY,self.config.C, dtype=torch.float64)
        init_state[0,X:X+self.init_size,Y:Y+self.init_size,0]=self.initialization.init
        if(self.config.C>1):
          init_state[0,:,:,1]=self.init_wall
        self.state = init_state.to(self.device)
        self.step_idx = 0


    def update_initialization_parameters(self):
        new_initialization_parameters = deepcopy(self.initialization_parameters)
        new_initialization_parameters['init'] = self.initialization.init.data
        if not self.initialization_space.contains(new_initialization_parameters):
            new_initialization_parameters = self.initialization_space.clamp(new_initialization_parameters)
            warnings.warn('provided parameters are not in the space range and are therefore clamped')
        self.initialization_parameters = new_initialization_parameters

    def update_update_rule_parameters(self):
        new_update_rule_parameters = deepcopy(self.update_rule_parameters)
        #gather the parameter from the lenia step (which may have been optimized)
        new_update_rule_parameters['m'] = self.lenia_step.m.data
        new_update_rule_parameters['s'] = self.lenia_step.s.data
        new_update_rule_parameters['r'] = self.lenia_step.r.data
        new_update_rule_parameters['rk'] = self.lenia_step.rk.data
        new_update_rule_parameters['b'] = self.lenia_step.b.data
        new_update_rule_parameters['w'] = self.lenia_step.w.data
        new_update_rule_parameters['h'] = self.lenia_step.h.data
        if not self.update_rule_space.contains(new_update_rule_parameters):
            new_update_rule_parameters = self.update_rule_space.clamp(new_update_rule_parameters)
            warnings.warn('provided parameters are not in the space range and are therefore clamped')
        self.update_rule_parameters = new_update_rule_parameters

    def step(self, intervention_parameters=None):
        self.state = self.lenia_step(self.state)
        self.step_idx += 1
        return self.state


    def forward(self):
        state = self.step(None)
        return state


    def run(self):
        """ run lenia for the number of step specified in the config.
        Returns the observations containing the state at each timestep"""
        #clip parameters just in case
        if not self.initialization_space['init'].contains(self.initialization.init.data):
          self.initialization.init.data = self.initialization_space['init'].clamp(self.initialization.init.data)
        if not self.update_rule_space['r'].contains(self.lenia_step.r.data):
            self.lenia_step.r.data = self.update_rule_space['r'].clamp(self.lenia_step.r.data)
        if not self.update_rule_space['rk'].contains(self.lenia_step.rk.data):
            self.lenia_step.rk.data = self.update_rule_space['rk'].clamp(self.lenia_step.rk.data)
        if not self.update_rule_space['b'].contains(self.lenia_step.b.data):
            self.lenia_step.b.data = self.update_rule_space['b'].clamp(self.lenia_step.b.data)
        if not self.update_rule_space['w'].contains(self.lenia_step.w.data):
            self.lenia_step.w.data = self.update_rule_space['w'].clamp(self.lenia_step.w.data)
        if not self.update_rule_space['h'].contains(self.lenia_step.h.data):
            self.lenia_step.h.data = self.update_rule_space['h'].clamp(self.lenia_step.h.data)
        if not self.update_rule_space['m'].contains(self.lenia_step.m.data):
            self.lenia_step.m.data = self.update_rule_space['m'].clamp(self.lenia_step.m.data)
        if not self.update_rule_space['s'].contains(self.lenia_step.s.data):
            self.lenia_step.s.data = self.update_rule_space['s'].clamp(self.lenia_step.s.data)
        # self.generate_init_state()
        observations = Dict()
        observations.timepoints = list(range(self.config.final_step))
        observations.states = torch.empty((self.config.final_step, self.config.SX, self.config.SY,self.config.C))
        observations.states[0]  = self.state
        for step_idx in range(1, self.config.final_step):
            cur_observation = self.step(None)
            observations.states[step_idx] = cur_observation[0,:,:,:]


        return observations

    def save(self, filepath):
        """
        Saves the system object using torch.save function in pickle format
        Can be used if the system state's change over exploration and we want to dump it
        """
        torch.save(self, filepath)


    def close(self):
        pass

