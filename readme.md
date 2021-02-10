# Robot Walk 
This is a dynamic program for finding the value function and hence optimal policy.  
This uses parallel programming using CUDA/Numba. This program also generates the value gradient in bmp format in gray scale.

## Requirements
Python&nbsp; 3.7  
CUDA &nbsp; 11.0

## Dependencies
> imageio&emsp;&emsp;&emsp;&emsp;&emsp;2.9.0  
> numba&nbsp; &emsp;&emsp;&emsp;&emsp;&emsp;0.51.2   
> numpy&nbsp; &emsp;&emsp;&emsp;&emsp;&emsp;1.19.2  
> scipy&nbsp;&nbsp; &nbsp; &emsp;&emsp;&emsp;&emsp;&emsp;1.5.2  

## Execution
python robot-walk.py > debug.out 2>&1 

