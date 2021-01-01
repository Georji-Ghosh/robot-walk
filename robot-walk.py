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

import agent
import time
import os
import shutil
import random


def build(map):
    if os.path.exists("./solution/{}".format(map)):
        shutil.rmtree("./solution/{}".format(map))
    os.makedirs("./solution/{}".format(map))

    start = time.time()
    robot = agent.Agent("./maps/{}.bmp".format(map))
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{} time taken for generating environment {:0>2}:{:0>2}:{:05.2f}".format(map, int(hours), int(minutes), seconds))
    robot.print("./solution/{}/values.dat".format(map))
    robot.draw_values("./solution/{}/values.bmp".format(map))
    drops = []
    for _ in range(128):
        
        p = (random.randint(1,1023), random.randint(1,1023))
        while p in drops:
            p = (random.randint(1,1024), random.randint(1,1024))
        drops.append(p)

        start = time.time()
        robot.draw_moves(p, "./solution/{}/scenario_{}_{}.bmp".format(map,p[0],p[1]))
        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("{} time taken for finding moves for droping point {} is {:0>2}:{:0>2}:{:05.2f}".format(map, p, int(hours),int(minutes),seconds))

def main():
    random.seed(round(time.time()*1000))
    for name in ["one", "two", "three", "four", "five"]:
        build(name)

if __name__ == "__main__":
    main()

