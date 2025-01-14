import numpy as np
import matplotlib.pyplot as plt
import scipy
from generate_pointcloud.gen_cloud_GUI import GenereatePointCloud

# create the object to generate the point cloud using the GUI
gen_cloud = GenereatePointCloud()
gen_cloud.create_GUI()
gen_cloud.from_extern = True

# parameters of the analysis
start = 10
end = 100
step = 10
num_runs = 10

samples = np.arange(start, end+step, step)

# results obtained using our method
results = np.zeros((samples.shape[0], num_runs))

print("Generating point cloud using the proposed method:", end="")
for i in range(0,samples.shape[0]):
    gen_cloud.num_samples = samples[i]
    print("\n"+str(samples[i]), end=": ", flush=True)

    for j in range(0,num_runs):
        print(j, end=" ", flush=True)
        gen_cloud.generate_point_cloud()
        results[i][j] = gen_cloud.gen_time

avg_results = np.average(results, axis=1)
var_results = np.var(results, axis=1)

scipy.io.savemat("results/ur5e_gen_times.mat", {"samples": samples, 
                                        "avg_times": avg_results,
                                        "var_times": var_results})   

# create the plot
plt.scatter(x=samples, y=avg_results, marker="o")
plt.plot(samples, avg_results)
plt.grid(visible=True, alpha=0.5)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel(r"$Number\; of\; samples$")
plt.xlim(samples[0]-step, samples[-1]+step)
plt.xticks(samples)
plt.ylabel(r"$Avg\; generation\; time\; [s]$")
plt.show()


