# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Mon Jul 15 12:41:57 2013

This package contains the definition of all the abstraction patterns and
procedures related with the segmentation procedure we have selected, based
on a wavelet transformation on the signal.

@author: T. Teijeiro
"""

from .deflection import generate_Deflection_Patterns
from .rdeflections import RDEFLECTION_PATTERN