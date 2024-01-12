# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 00:23:13 2023

@author: skynet
"""
import sys
import os

def resourcepath(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)