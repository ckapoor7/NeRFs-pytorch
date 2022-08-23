from yacs.config import CfgNode as CN


_C = CN()

# system
_C.SYSTEM = CN()

# encoder
_C.ENCODERS = CN()
# encoder params
_C.ENCODERS.DIM_INPUT = 3
_C.ENCODERS.NUM_FREQS = 10
_C.ENCODERS.LOG_SPACE = True
_C.ENCODERS.USE_VIEWDIRS = True
_C.ENCODERS.NUM_FREQS_DIRS = 4

# stratified sampling
_C.STRAT_SAMPLING = CN()
# stratified sampling params
_C.STRAT_SAMPLING.NUM_SAMPLES = 64
_C.STRAT_SAMPLING.PERTURB = True
_C.STRAT_SAMPLING.INVERSE_DEPTH = True

# model
_C.MODEL = CN()
# model params
_C.MODEL.DIM_FILTER = 128
_C.MODEL.NUM_LAYERS = 2
_C.MODEL.SKIP = []
_C.MODEL.USE_FINE_MODEL = True
_C.MODEL.DIM_FILTERS_FINE = 128
_C.MODEL.NUM_LAYERS_FINE = 6

# hierarchical sampling
_C.H_SAMPLING = CN()
# hierarchical sampling params
_C.H_SAMPLING.NUM_HIERARCHICAL_SAMPLES = 64
_C.H_SAMPLING.PERTURB_HIERARCHICAL = False

# optimizer
_C.OPTIMIZER = CN()
# optimizer params
_C.OPTIMIZER.LR = 5e-4

# training
_C.TRAINING = CN()
# training params
_C.TRAINING.DATASET = "/path/to/dataset"
_C.TRAINING.SAVE_PATH = "/path/to/save"
_C.TRAINING.NUM_ITERS = 10000
_C.TRAINING.BATCH_SIZE = 2**14
_C.TRAINING.BATCHING_DISABLE = True
_C.TRAINING.CHUNKSIZE = 2**14
_C.TRAINING.CENTER_CROP = True
_C.TRAINING.CENTER_CROP_ITERS = 50
_C.TRAINING.DISPLAY_RATE = 25


def get_cfg_defaults():
    """
    get a YACS CfgNode object with default parameters
    """
    return _C.clone()
