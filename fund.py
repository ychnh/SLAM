import numpy as np
import math
from numpy.linalg import inv
def unorm_H(H,T):
    return inv(T)@H@T

# Fundamental Matrix using contraint in P. Normalized 8 point algorithm
# Constraint x`Fx = 0
def Fund_A_p(p, p_):
    x ,y ,_ = p[0]
    x_,y_,_ = p_[0]
    A = [[x_*x, x_*y, x_, y_*x, y_*y, y_, x, y, 1]]
    return np.array(A)
    
    
def Fund_A_P(P,P_):
    '''Construct a matrix A for calculating fund matrix'''
    As = []
    for p,p_ in zip(P,P_):
        As += [ Fund_A_p(p,p_) ]
    A = np.concatenate( As, axis=0 )
    return A

def Fund(P,P_):
    ''' Need 8 or more corresponding points '''
    A = Fund_A_P( P, P_ )
    f = solve_homogenous_Ax(A)
    F = vec_mat(f)
    return F

def Fund_loss(F, p, p_):
    # TODO: Also want to return epipolar lines using t
    # TODO: find the point on these lines that minize distance between the lines in the origin\
    # TODO: project 3d poiont X using these predicted image points on the lines
    # 12.5.2 Multiple View Geometry
    # Algorithm proceeds as follows
    # Transform image coordinates so that the point of interest lie at origin
    # find epipolar line wrt t such that distance is minimized

    tx,ty,_ = p[0]
    tx_,ty_,_ = p_[0]
    
    T = np.array([[1, 0, -tx],
                  [0, 1, -ty],
                  [0, 0, 1]])
    T_ = np.array([[1, 0, -tx_],
                   [0, 1, -ty_],
                   [0, 0, 1]])
    
    F = inv(T_.T) @ F @ inv(T)

    # e and e_ are solutions to F@e = 0 and e_@F=0 and will solve using SVD
    # "However, due to noise, F’ may not be singular. So instead, next best thing is eigenvector associated with smallest eigenvalue of F" (http://www.cse.psu.edu/~rtc12/CSE486/lecture19_6pp.pdf)
    e = solve_homogenous_Ax(F)
    e = e/e[2]
    # e_@F=0 we want to write this as a right multiplication and it is F.T @ e_ = 0
    e_= solve_homogenous_Ax(F.T)
    e_ = e_/e_[2]
    
    #now scale e so that e1^2 + e2^2 = 1 to construct the rotation matrix
    k = math.sqrt( 1/(e[0]**2 + e[1]**2))
    k_ = math.sqrt( 1/(e_[0]**2 + e_[1]**2))
    e = k*e
    e_ = k_*e_
    
    R = np.array([[  e[0], e[1], 0 ],
                  [ -e[1], e[0], 0 ],
                  [     0,    0, 1 ]])
    R_ = np.array([[  e_[0], e_[1], 0 ],
                  [ -e_[1], e_[0], 0 ],
                  [     0,    0, 1 ]])
    
    F = R@F@R.T
    
    # Construction equation to minimizae distance
    f, f_ = e[2], e_[2]
    a, b, c, d = F[2,2], F[2,3], F[3,2], F[3,3]
    # roots of  t((at+b)^2 +p^2(ct+d)^2 )^2-(ad-bc)(1+f^2t^2)^2(at+b)(ct+d)  (eq12.7) #f_ is p here
    # numpy.roots()
    # The algorithm relies on computing the eigenvalues of the companion matrix
    # Use the Collect( eq, t) in Wolfram
    # t^3 (6 a^2 b^2 + 2 a^2 d^2 f_^2 + 2 b c f^2 (b c - a d) + 2 a d f^2 (b c - a d) + 8 a b c d f_^2 + 2 b^2 c^2 f_^2 + 6 c^2 d^2 f_^4) + 
    # t^5 (a^4 + 2 a^2 c^2 f_^2 + a d f^4 (b c - a d) + b c f^4 (b c - a d) + c^4 f_^4) +
    # t^4 (4 a^3 b + 4 a^2 c d f_^2 + 4 a b c^2 f_^2 + b d f^4 (b c - a d) + 2 a c f^2 (b c - a d) + 4 c^3 d f_^4) + 
    # t (b c (b c - a d) + a d (b c - a d) + b^4 + 2 b^2 d^2 f_^2 + d^4 f_^4) + 
    # t^2 (4 a b^3 + 2 b d f^2 (b c - a d) + a c (b c - a d) + 4 a b d^2 f_^2 + 4 b^2 c d f_^2 + 4 c d^3 f_^4) + 
    # t^6 a c f^4 (b c - a d) + 
    # b d (b c - a d)
    coeff_eq = [ a * c * f**4 * (b * c - a * d), #t^6
                (a**4 + 2 * a**2 * c**2 * f_**2 + a * d * f**4 * (b * c - a * d) + b * c * f**4 * (b * c - a * d) + c**4 * f_**4), #t^5
                (4 * a**3 * b + 4 * a**2 * c * d * f_**2 + 4 * a * b * c**2 * f_**2 + b * d * f**4 * (b * c - a * d) + 2 * a * c * f**2 * (b * c - a * d) + 4 * c**3 * d * f_**4), #t^4
                (6 * a**2 * b**2 + 2 * a**2 * d**2 * f_**2 + 2 * b * c * f**2 * (b * c - a * d) + 2 * a * d * f**2 * (b * c - a * d) + 8 * a * b * c * d * f_**2 + 2 * b**2 * c**2 * f_**2 + 6 * c**2 * d**2 * f_**4), #t^3
                (4 * a * b**3 + 2 * b * d * f**2 * (b * c - a * d) + a * c * (b * c - a * d) + 4 * a * b * d**2 * f_**2 + 4 * b**2 * c * d * f_**2 + 4 * c * d**3 * f_**4), #t^2
                (b * c * (b * c - a * d) + a * d * (b * c - a * d) + b**4 + 2 * b**2 * d**2 * f_**2 + d**4 * f_**4), #t^1
                b * d * (b * c - a * d)] #t^0
    real_roots = np.real( np.roots(coeff_eq) )
    def sq_dist(t, f,f_, a,b,c,d):
        return (t**2)/(1 + f**2*t**2) + ( (c*t+d)**2 )/( (a*t+b)**2 + f_**2*(c*t+d)**2 )
    
    sq_dists = [sq_dist(t,f,f_,a,b,c,d) for t in real_roots]
    min_t = np.argmin(sq_dists)
    return sq_dists[min_t]
    
def norm_RANSAC_Fund(iterations, loss_tresh, num_outliers_tresh, Pts, Pts_):

    def inliers_outliers(losses, loss_tresh,  corr):
        ''' get inlier/outliers/total loss/'''
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
        sample = random.sample( corr, 8)
        P,P_ = zip(*sample)
        F = Fund(P, P_)
        losses = [Fund_loss(F, p,p_) for p,p_ in corr]
        inliers, outliers = inliers_outliers(losses, loss_tresh, corr)
        if len(outliers) < num_outliers_tresh:
            P,P_ = zip(*inliers)
            F = Fund(P,P_)
            losses = [Fund_loss(F, p,p_) for p,p_ in corr]
            total_loss = sum(losses)
            if total_loss < best_loss:
                best_model = F

        #select 4 random points and fit H
        # compute losses and find outliers
        # if num outlier < nO
            #refit model using all inliers
            #compute losses and compare to best_score

    return inv(tfN_) @ F @ tfN
