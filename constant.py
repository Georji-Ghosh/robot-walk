
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

IMAGE_DIM   = (1024, 1024, 3)
DATA_DIM    = (1024, 1024)
THREADS     = 1024
BLOCKS      = 1

LAND = 0          # WHITE
BLOCK = 2         # BLACK
WATER = 4         # BLUE
GOAL = 1          # GREEN
FIRE = 8          # RED

MAX_MOVES   =  1024 * 10
STEPS       =  1024 * 10
LAND_VALUE  = -1             # WHITE
BLOCK_VALUE = -1*10*STEPS    # BLACK
WATER_VALUE = -100           # BLUE
FIRE_VALUE  = -250           # FIRE 
GOAL_VALUE  =  0             # GREEN

