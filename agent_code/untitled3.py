# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:13:39 2022

@author: isabe
"""
import numpy as np
field= np.array([[1,-1,-1],[1,1,0],[8,1,0]])
a= 1
b= 1
surroundings =[1 if (field[a,b-1]==-1 or field[a,b-1]==1) else 0,
                    1 if (field[a+1,b]==-1 or field[a+1,b]==1) else 0, 
                    1 if (field[a,b+1]==-1 or field[a,b+1]==1) else 0,
                    1 if (field[a-1,b]==-1 or field[a-1,b]==1) else 0]