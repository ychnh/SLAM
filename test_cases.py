import numpy as np
import core
from fund import *
from hom import *

def test_hom_1():
    a = np.random.random( (1,3) )
    b = np.random.random( (1,3) )
    c = np.random.random( (1,3) )
    d = np.random.random( (1,3) )
    A = Hom_A_P( [a,b,c,d],[a,b,c,d] )
    print('A',A.round(2))
    h = core.solve_homogenous_Ax(A).round(2)
    H = core.vec_mat(h)
    print('Test Hom 1: should print scale-equivalent identity')
    print(H)
    
def test_fund_1():
    a = np.random.random( (1,3) )
    b = np.random.random( (1,3) )
    c = np.random.random( (1,3) )
    d = np.random.random( (1,3) )

    A = Fund_A_P( [a,b,c,d,a,b,c,d],[a,b,c,d,a,b,c,d] )
    print('A\n',A.round(2))
    f = core.solve_homogenous_Ax(A).round(2)
    F = core.vec_mat(f)
    print('Test Fund 1: x_Fx should be zero')
    x = np.random.random( (1,3) )
    print('F\n',F.round(2))
    print( 'xFx\n',x@F@x.T )

