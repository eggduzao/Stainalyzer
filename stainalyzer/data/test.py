import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy.stats import f
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import OrderedDict, Counter
from typing import Generator, List, Any, Callable, Dict, Tuple, Optional

def run():
    def count():

        proportions_dict = {
            1: [96, 3, 1, 0, 0],
            2: [93, 4, 2, 1, 0],
            3: [90, 5, 3, 2, 0],
            4: [87, 7, 3, 3, 0],
            5: [84, 7, 4, 4, 1],
            6: [81, 8, 5, 5, 1],
            7: [78, 10, 5, 6, 1],
            8: [75, 11, 6, 7, 1],
            9: [72, 12, 6, 8, 2],
            10: [69, 13, 7, 9, 2],
            11: [66, 14, 8, 10, 2],
            12: [63, 15, 9, 11, 2],
            13: [60, 16, 9, 12, 3],
            14: [57, 18, 9, 13, 3],
            15: [54, 18, 11, 14, 3],
            16: [48, 19, 15, 15, 3],
            17: [45, 19, 16, 16, 4],
            18: [43, 20, 16, 17, 4],
            19: [42, 20, 16, 18, 4],
            20: [40, 21, 17, 18, 4],
            21: [39, 21, 17, 19, 4],
            22: [36, 22, 17, 20, 5],
            23: [35, 22, 18, 20, 5],
            24: [34, 21, 19, 21, 5],
            25: [32, 21, 19, 22, 6],
            26: [30, 20, 19, 24, 7],
            27: [27, 20, 20, 26, 7],
            28: [25, 20, 21, 27, 7],
            29: [24, 19, 21, 28, 8],
            30: [23, 18, 22, 29, 8],
            31: [22, 17, 23, 29, 9],
            32: [21, 16, 24, 30, 9],
            33: [18, 15, 26, 31, 10],
            34: [15, 14, 30, 31, 10],
            35: [12, 13, 32, 32, 11],
            36: [10, 12, 33, 33, 12],
            37: [9, 11, 34, 34, 12],
            38: [6, 11, 34, 35, 14],
            39: [4, 11, 36, 35, 14],
            40: [3, 10, 37, 35, 15],
            41: [2, 9, 38, 36, 15],
            42: [0, 9, 39, 36, 16],
            43: [0, 8, 39, 37, 16],
            44: [0, 7, 38, 38, 17],
            45: [0, 7, 37, 39, 17],
            46: [0, 6, 36, 40, 18],
            47: [0, 6, 35, 41, 18],
            48: [0, 5, 34, 42, 19],
            49: [0, 5, 33, 43, 19],
            50: [0, 5, 32, 44, 19],
            51: [0, 4, 32, 44, 20],
            52: [0, 3, 32, 45, 20],
            53: [0, 3, 32, 45, 20],
            54: [0, 3, 31, 46, 20],
            55: [0, 2, 31, 46, 21],
            56: [0, 2, 30, 47, 21],
            57: [0, 1, 29, 48, 22],
            58: [0, 1, 28, 49, 22],
            59: [0, 0, 25, 52, 23],
            60: [0, 0, 24, 53, 23],
            61: [0, 0, 23, 54, 23],
            62: [0, 0, 22, 54, 24],
            63: [0, 0, 22, 53, 25],
            64: [0, 0, 21, 54, 25],
            65: [0, 0, 20, 54, 26],
            66: [0, 0, 19, 55, 26],
            67: [0, 0, 18, 55, 27],
            68: [0, 0, 17, 56, 27],
            69: [0, 0, 16, 56, 28],
            70: [0, 0, 15, 57, 28],
            71: [0, 0, 13, 58, 29],
            72: [0, 0, 12, 59, 29],
            73: [0, 0, 11, 59, 30],
            74: [0, 0, 10, 60, 30],
            75: [0, 0, 9, 60, 31],
            76: [0, 0, 8, 61, 31],
        }



        for key, value in proportions_dict.items():
            s = sum(value)
            print(f"{key} = {s}")



    count()

if __name__ == "__main__":
    run()