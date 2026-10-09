"""MaPLe factory for the H02 reference encoder."""
import copy
import logging

from .maple import load as maple_model

logger = logging.getLogger('mylogger')


def get_model(model_dict, init_classname=None, verbose=False):
    if model_dict.get('arch') != 'maple':
        raise ValueError('H02 reference training requires arch=maple')
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop('arch')
    model = maple_model(param_dict, init_classname)
    if verbose:
        logger.info(model)
    return model
