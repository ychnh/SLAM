import numpy as np
import math
# For sake of clarity of matrix operations, all points and vectors will be represented as matrices
# instead of the traditional list form.
# This means a point [x,y,z] will be expressed as [[x,y,z]] to denote a 1x3 matrix


def solve_homogenous_Ax(A):
    ''' With constraint that ||x||=1 '''
    U,S,V_trans = np.linalg.svd(A)
    V = V_trans.transpose()
    return V[:,-1] # Last column of V

def vec_mat(vec):
    '''given a 1x9 vector converts to 3x3 matrix'''
    M1 = vec[0:3].reshape( (1,3) )
    M2 = vec[3:6].reshape( (1,3) )
    M3 = vec[6:9].reshape( (1,3) )
    return np.concatenate( [M1, M2, M3], axis=0)

def p_rep(p):
    ''' homogeneous representation of a coordinate '''
    k = p[0][2]
    return p/k

def dist(p, p_):
        p = p_rep(p)
        p_ = p_rep(p_)

        x,y,_ = p[0]
        x_,y_,_ = p_[0]

        return math.sqrt( (x-x_)**2 + (y-y_)**2 )


# Normalization
# Only modifying (x,y) move centroid of all points to origin
# We choose to scale the coordinates so that the average distance of a point x from the origin is sqrt(2) Scale x,y together

def normalize_and_get_transform(P):

    def get_centroid(P):
        sum_P = np.zeros( (1,3) )
        for p in P:
            sum_P += p
        return sum_P/len(P)

    tx, ty, _ = get_centroid(P) # we will translate by -x,-y of centeroid
    T_trans = np.array( [[1,0, -tx],
                         [0,1,-ty],
                         [0,0,1]])
    centered_P = []
    for p in P:
        centered_P += [ T_trans@p.T ]
    
        
    z = np.array([[0,0,1]])
    dist_sum = 0
    for p in centered_P:
        dist_sum += dist(p,z)
    avg_dist = dist_sum/len(P)
    k = math.sqrt(2) / avg_dist
    
    T_scale = np.array( [[k,0,0],
                         [0,k,0],
                         [0,0,1]])
    
    scaled_centered_P = []
    for p in centered_P:
        scaled_centered_P += [ T_scale@p.T]
    return scaled_centered_P, T_scale@T_trans
