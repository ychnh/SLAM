import numpy as np
import sys
import random
import math
from numpy.linalg import inv
import core

# Homography using constraint in P2: p_ x Hp = 0
# A cross product with itself is zero
def Hom_A_p(p, p_):
    x,y,w = p_[0]
    Z = np.array([[0,0,0]])
    A_1 = np.concatenate( [Z, -w*p, y*p], axis=1)
    A_2 = np.concatenate( [w*p, Z, -x*p], axis=1)
    #A_3 = np.concatenate( [-y*p, x*p, Z], axis=1) Omit third linear eq. since dependant on first two due to scale
    A = np.concatenate( [A_1, A_2], axis=0)
    return A

def Hom_A_P(P, P_):
    ''' Constructs the system of equations to compute (H)omography with 4 corresponding points'''
    As = []
    for p,p_ in zip(P,P_):
        As += [ Hom_A_p(p,p_) ]
    A = np.concatenate( As, axis=0 )
    return A

def Hom(P, P_):
    ''' Need 4 or more corresponding points '''
    A = Hom_A_P( P, P_ )
    h = solve_homogenous_Ax(A)
    H = vec_mat(h)
    return H

# symmetric transfer error Multiview Geometry 4.8
def Hom_loss(H, p, p_):
    Hp_ = H @ p
    Hp = inv(H) @ p_
    return core.dist(Hp_,p_) + core.dist(Hp, p)


def norm_RANSAC_Hom(iterations, loss_tresh, num_outliers_tresh, Pts, Pts_):
    ''' Normed DLT 4.4 Ransac 4.7 '''

    def inliers_outliers(losses, loss_tresh,  corr):
        ''' get inlier/outliers/total loss/'''
        #TODO: Check speed of this algo
        T = loss_tresh
        inliers = [ corr[i] for i,L in enumerate(losses) if L<T ]
        outliers = [ corr[i] for i,L in enumerate(losses) if L>=T ]
        return inliers, outliers


    nPts , tfN = core.normalize_and_get_transform(Pts)
    nPts_, tfN_= core.normalize_and_get_transform(Pts_)
    corr = zip(nPts, nPts_)

    best_loss = sys.float_info.max
    best_model = None
    for i in range(iterations):
        sample = random.sample( corr, 4)
        P,P_ = zip(*sample)
        H = Hom(P, P_)
        losses = [Hom_loss(H, p,p_) for p,p_ in corr]
        inliers, outliers = inliers_outliers(losses, loss_tresh, corr)
        if len(outliers) < num_outliers_tresh:
            P,P_ = zip(*inliers)
            H = Hom(P,P_)
            losses = [Hom_loss(H, p,p_) for p,p_ in corr]
            total_loss = sum(losses)
            if total_loss < best_loss:
                best_model = H

        #select 4 random points and fit H
        # compute losses and find outliers
        # if num outlier < nO
            #refit model using all inliers
            #compute losses and compare to best_score

    return inv(tfN_) @ H @ tfN
