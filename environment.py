# Copyright (c) 2020 Georji Indranil Ghosh <georji.ghosh@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from os import path
import imageio
import numpy as np
import numba
from numba import cuda

from constant import *


class Environment(object):

    def __init__(self, image=None):
        self.map = np.zeros(DATA_DIM, dtype=np.uint8)

        if image == None:
            return
        self.build(image)

    def __build_using_cuda__(self):
       
        @cuda.jit(device=True)
        def __compare_pixel__(pixel, r, b, g):
            if (pixel[0] == r and 
                pixel[1] == b and
                pixel[2] == g):
                return True
            else:
                return False

        @cuda.jit(device=True)
        def __build_map_object__(pixel):
            if __compare_pixel__(pixel, 255, 0, 0):               # RED
                return FIRE
            elif __compare_pixel__(pixel, 255, 255, 255):         # WHITE
                return LAND
            elif __compare_pixel__(pixel, 0, 0, 0):               # BLACK
                return BLOCK
            elif __compare_pixel__(pixel, 0, 255, 0):             # GREEN
                return GOAL
            elif __compare_pixel__(pixel, 0, 0, 255):             # BLUE
                return WATER
            else:
                return LAND

        @cuda.jit
        def __build_map__(pixels, map):
            thread_id = cuda.threadIdx.x
            for y in range(DATA_DIM[1]):
                map[thread_id][y] = __build_map_object__(pixels[thread_id][y])
        
        __build_map__[BLOCKS, THREADS](self.pixels, self.map)

    def build(self, image):
        if not path.exists(image):
            raise FileNotFoundError("Map image doesn't exists")

        self.pixels = imageio.imread(image)
        if self.pixels.shape != IMAGE_DIM:
            raise AttributeError(
                "Incorrect image dimention. It must be {}".format(IMAGE_DIM))
        self.__build_using_cuda__()

    def print(self):
        for x in range(DATA_DIM[0]):
            for y in range(DATA_DIM[1]):
                print(self.map[x][y], end= "")
            print("")
        print("")
