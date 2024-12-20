from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

from .datasets.mot import MOT

dataset_factory = {
  'mot': MOT
}


def get_dataset(dataset):
  return dataset_factory[dataset]
