#!/usr/bin/env python3
import sys
sys.path.append("..")

import numpy as np
import math
from pymoo.core.problem import ElementwiseProblem

class EllipsoidEquationOptProblem(ElementwiseProblem):
    def __init__(self, *args):
        # retrieve the set of points
        self.center = args[0]
        self.points = args[1]
        self.num_points = self.points.shape[0]
        self.viz_res = args[2]
        self.num_points_wt = args[3]
        self.volume_wt = args[4]

        # retrieve the passed center
        if self.center is None:
            # compute an estimation of the center
            xc = np.mean([np.min(self.points[:,0]), np.max(self.points[:,0])])
            yc = np.mean([np.min(self.points[:,1]), np.max(self.points[:,1])])
            zc = np.mean([np.min(self.points[:,2]), np.max(self.points[:,2])])
            self.center = np.array([xc, yc, zc])

        if self.viz_res==True:
            print("Center estimation: {:.4f}, {:.4f}, {:.4f}".format(self.center[0], self.center[1], self.center[2]))

        super().__init__(n_var = 6,
                         n_obj = 1,
                         n_ieq_constr = 3,
                         xl = [pow(10,-4), pow(10,-4), pow(10,-4), -10, -10, -10],
                         xu = 10)

    def _evaluate(self, x, out, *args, **kwargs):
        # retrieve the equation paramters
        a = x[0]
        b = x[1]
        c = x[2]

        # retrieve the approximation of the center
        xc = x[3]
        yc = x[4]
        zc = x[5]

        # retrieve the points composing the pointcloud
        points = self.points

        # count the number of points inside the actual equation
        xp, yp, zp = points[:,0], points[:,1], points[:,2]
        val = ((xp-xc)**2)/(a**2) + ((yp-yc)**2)/(b**2) + ((zp-zc)**2)/(c**2) - 1

        inner_points = np.sum(val <= 0)
        
        # constraints definition
        constrs = [-x[0], -x[1], -x[2]]
        out["G"] = np.row_stack(constrs)
    
        # objective function definition
        out["F"] = -self.num_points_wt*inner_points + self.volume_wt*4/3*np.pi*a*b*c