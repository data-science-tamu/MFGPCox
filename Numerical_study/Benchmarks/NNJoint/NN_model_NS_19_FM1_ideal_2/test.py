import numpy as np
import matplotlib.pyplot as plt


s = np.load("./survivals_probs_loss2_t50_ndata50/sp_54.npy")


#
# est_sz_mean = []
# est_sz_lower = []
# est_sz_upper = []

est_sz_mean = np.mean(s, axis=0)
est_sz_lower = np.quantile(s, 0.025, axis=0)
est_sz_upper = np.quantile(s, 0.975, axis=0)

import matplotlib.pyplot as plt

time_steps = range(50, 50 + len(s[0]))

# plt.plot(time_steps, true_sz[0], label="true", color="orange")
plt.plot(time_steps, est_sz_mean, label="Mean Survival Estimate", color="blue")
# plt.fill_between(time_steps, est_sz_lower[0], est_sz_upper[0], color="blue", alpha=0.2, label="95% Confidence Interval")
plt.plot(time_steps, est_sz_lower, label="interval", color="black", linestyle="--")
plt.plot(time_steps, est_sz_upper, label="interval", color="black", linestyle="--")

plt.xlabel("Time Steps")
plt.ylabel("Survival Estimate")
plt.title("Survival Estimate with Confidence Interval")
plt.legend()
plt.show()
