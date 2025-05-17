"""
Music source separation tool for isolating vocals and instruments
Author: Roman Solovyev (ZFTurbo)
GitHub: https://github.com/ZFTurbo/
"""

__version__ = '0.1.0'
__author__ = 'Roman Solovyev (ZFTurbo)'

from .inference import proc_folder

__all__ = ['proc_folder']
