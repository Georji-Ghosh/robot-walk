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

import imageio
import numpy as np
import numba
from numba import cuda
import copy

import environment
from constant import *

class Agent(object):

    def __init__(self, map=None):
        self.value = np.zeros(DATA_DIM, dtype=np.int32)
        if map == None:
            return
        self.build(environment.Environment(map))


    def build(self, env):
        self.env = env
        tarnsition = self.__init_using_cuda__()
        self.__update_using_cuda__(tarnsition)

    def __init_using_cuda__(self):

        @cuda.jit(device=True)
        def __build_value_object__(obj):
            if obj == LAND:
                return LAND_VALUE
            if obj == GOAL:
                return GOAL_VALUE
            if obj == BLOCK:
                return BLOCK_VALUE
            if obj == WATER:
                return WATER_VALUE
            if obj == FIRE:
                return FIRE_VALUE
            return LAND_VALUE
            
        @cuda.jit
        def __init_value__(map, value, tarnsition):
            thread_id = cuda.threadIdx.x
            for y in range(DATA_DIM[1]):
                tarnsition[thread_id][y] = __build_value_object__(map[thread_id][y])
                value[thread_id][y] = 0
        
        tarnsition = np.zeros(DATA_DIM, dtype=np.int32)
        __init_value__[BLOCKS, THREADS](self.env.map, self.value, tarnsition)

        return tarnsition

    def __update_using_cuda__(self, tarnsition):

        @cuda.jit(device=True)
        def __calculate_value__(value, x, y, reward):
            if reward >= GOAL:
                return reward
            max_value = value[x-1][y] if x-1 >= 0 else value[x+1][y]
            max_value = max(max_value, value[x-1][y]) if x-1 >= 0 else max_value
            max_value = max(max_value, value[x+1][y]) if x+1 < DATA_DIM[0] else max_value
            max_value = max(max_value, value[x][y-1]) if y-1 >= 0 else max_value
            max_value = max(max_value, value[x][y+1]) if y+1 < DATA_DIM[1] else max_value
            return reward + max_value
        
        @cuda.jit
        def __update_value__(cache, value, tarnsition, iteration):
            thread_id = cuda.threadIdx.x
            for _ in range(iteration):
                for y in range(DATA_DIM[1]):
                    cache[thread_id][y] = __calculate_value__(value, thread_id, y, tarnsition[thread_id][y])
                
                cuda.syncthreads()
                for y in range(DATA_DIM[1]):
                    value[thread_id][y] = cache[thread_id][y]
                cuda.syncthreads()
        
        __update_value__[BLOCKS, THREADS](np.zeros(DATA_DIM, dtype=np.int32), self.value, tarnsition, STEPS)


    def print(self, file):
        self.__dump_data__(self.value, file)

    def __dump_data__(self, data, file):
        f = open(file, "w")
        lines = []
        for x in range(DATA_DIM[0]):
            line = ""
            for y in range(DATA_DIM[1]):
                line += "{:>+10d} ".format(data[x][y]) 
            line +="\n"
            lines.append(line)
        f.writelines(lines)
        f.close()

    def draw_moves(self, p, file):
        image = copy.deepcopy(self.env.pixels)
        goal = self.value.max()
        low = self.value.min()
        track= []
        while self.value[p[0]][p[1]] != goal and len(track) < MAX_MOVES:
            p1 = self.__update_xy__(p, low - 1, track)
            if p == p1:
                print("Unable to solve: in loop")
                break
            p = p1
            image[p[0]][p[1]][0] = 255
            image[p[0]][p[1]][1] = 0
            image[p[0]][p[1]][2] = 0
            track.append(p)
        imageio.imwrite(file, image)

    def __update_xy__(self, p, lowest, track):
        cadidates = []
        if p[0] - 1 >= 0 and (p[0] - 1, p[1]) not in track:
            cadidates.append((p[0] - 1, p[1]))
        if p[0] + 1 < DATA_DIM[0] and (p[0] + 1, p[1]) not in track:
            cadidates.append((p[0] + 1, p[1]))
        if p[1] - 1 >= 0 and (p[0], p[1] - 1) not in track:
            cadidates.append((p[0], p[1] - 1))
        if p[1] + 1 < DATA_DIM[1] and (p[0], p[1] + 1) not in track:
            cadidates.append((p[0], p[1] + 1))
        
        p1 = p
        if len(cadidates) > 0:
            max_val = lowest
            for i in cadidates:
                if max_val <= self.value[i[0]][i[1]]:
                    max_val = self.value[i[0]][i[1]]
                    p1 = i
        return p1

    def draw_values(self, file):

        @cuda.jit
        def __update_image__(image, value, marker, min_val):
            thread_id = cuda.threadIdx.x
            for y in range(DATA_DIM[1]):
                val = value[thread_id][y] - min_val
                val = round(val/marker*25) if val < marker else round((val-marker)/STEPS*130)+25
                image[thread_id][y][0] = val
                image[thread_id][y][1] = val
                image[thread_id][y][2] = val
        
        image = copy.deepcopy(self.env.pixels)
        max_val = self.value.max()
        min_val = self.value.min()
        __update_image__[BLOCKS, THREADS](image, self.value, (max_val - min_val) - STEPS, min_val)
        imageio.imwrite(file, image)
