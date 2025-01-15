import os
import numpy as np
import h5py
import time
import matplotlib.pyplot as plt
from opt_problem.eqn_solv_opt import solve_eqn_prob
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from pymoo.termination.robust import RobustTermination
from pymoo.termination.ftol import SingleObjectiveSpaceTermination


if __name__=="__main__":
    # load the .h5 file
    h5file_path = "./data/evaluation_Dataset_1000.h5"
    h5_file = h5py.File(h5file_path, "r")

    # extract the equation parameters and the corresponding point clouds
    equation_params = h5_file.get("labels")[:,:].transpose((1,0))
    pointclouds = h5_file.get("train_dataset")[:,:,:].transpose((2,1,0))

    # init the result data structure
    results = np.zeros(equation_params.shape)
    results[:,(0,4)] = equation_params[:,(0,4)]

    equation_params = equation_params[:,(1,2,3,5,6,7)]
    num_elements = equation_params.shape[0]

    # init the error data structure:
    errors = np.zeros(equation_params.shape)
    times = np.zeros(equation_params.shape[0])

    for i in range(num_elements):
        # os.system("clear")
        print("Solved problems: {:3d}/{:3d}".format(i+1, num_elements), flush=True, end=" ")
        points = pointclouds[i,:,:]

        # solve the optimization problem
        start = time.time()
        # alg_name = "PatternSearch"
        # alg_name = "GA"
        alg_name = "PSO"
        res = solve_eqn_prob(points, alg_name)
        times[i] = (time.time() - start)
        res = res.X

        # store the obtained results
        results[i,(1,2,3,5,6,7)] = res

        # compute the relative error
        errors[i,0] = np.abs(res[0]-equation_params[i,0])/equation_params[i,0]
        errors[i,1] = np.abs(res[1]-equation_params[i,1])/equation_params[i,1]
        errors[i,2] = np.abs(res[2]-equation_params[i,2])/equation_params[i,2]
        errors[i,3] = np.abs(res[3]-equation_params[i,3])/np.abs(equation_params[i,3])
        errors[i,4] = np.abs(res[4]-equation_params[i,4])/np.abs(equation_params[i,4])
        errors[i,5] = np.abs(res[5]-equation_params[i,5])/np.abs(equation_params[i,5])
        print([f"{x:.4f}" for x in errors[i,:]*100])

    
    # compute different errors metrics for each predicted parameter
    mean_err = np.mean(errors, 0)*100
    median_err = np.median(errors,0)*100
    min_err = np.min(errors,0)*100
    max_err = np.max(errors,0)*100
    var_err = np.var(errors,0)

    # compute the formatted string
    formatted_mean_err = "\t".join(f"{err:08.4f}%" for err in mean_err)
    formatted_median_err = "\t".join(f"{err:08.4f}%" for err in median_err)
    formatted_min_err = "\t".join(f"{err:08.4f}%" for err in min_err)
    formatted_max_err = "\t".join(f"{err:08.4f}%" for err in max_err)

    print("\t\tA\t\tB\t\tC\t\tXc\t\tYc\t\tZc")
    print(f"Mean Error:\t{formatted_mean_err}")
    print(f"Median Error:\t{formatted_median_err}")
    print(f"Min Error:\t{formatted_min_err}")
    print(f"Max Error:\t{formatted_max_err}")

    # store all the results in a .h5 file
    res_h5_path = './results/optProb_'+alg_name+'_result_'+h5file_path.split('/')[-1]
    with h5py.File(res_h5_path, 'w') as h5file:
        # save the times
        h5file.create_dataset('times', data=times)
        # save the results to the file
        h5file.create_dataset('result', data=results)
        # save the errors to the file
        h5file.create_dataset('errors', data=errors)
        # save the error metrics
        h5file.create_dataset('mean_error', data=mean_err)
        h5file.create_dataset('median_error', data=median_err)
        h5file.create_dataset('min_error', data=min_err)
        h5file.create_dataset('max_error', data=max_err)

