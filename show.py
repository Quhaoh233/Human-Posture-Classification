# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 13:49:36 2021

@author: nkliu
"""

import numpy as np
from tool.visualise import visualise
from tool.graph import Graph

sample_path = 'data/train/002/P000S00G10B10H10UC022000LC021000A002R0_08241716.npy'
sample = np.load(sample_path)
visualise(sample, graph=Graph(), is_3d=False)