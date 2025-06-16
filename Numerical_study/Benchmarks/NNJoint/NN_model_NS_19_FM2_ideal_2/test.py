import numpy as np
import matplotlib.pyplot as plt

s = np.load("./survivals_probs_loss2_t20_ndata50/sp_113.npy")



est_sz_mean = []
est_sz_lower = []
est_sz_upper = []

est_sz_mean.append(np.mean(s, axis=0))
est_sz_lower.append(np.quantile(s, 0.025, axis=0))
est_sz_upper.append(np.quantile(s, 0.975, axis=0))

import matplotlib.pyplot as plt

time_steps = range(20, 20 + len(s[0]))

# plt.plot(time_steps, true_sz[0], label="true", color="orange")
plt.plot(time_steps, est_sz_mean[0], label="Mean Survival Estimate", color="blue")
# plt.fill_between(time_steps, est_sz_lower[0], est_sz_upper[0], color="blue", alpha=0.2, label="95% Confidence Interval")
plt.plot(time_steps, est_sz_lower[0], label="interval", color="black", linestyle="--")
plt.plot(time_steps, est_sz_upper[0], label="interval", color="black", linestyle="--")

plt.xlabel("Time Steps")
plt.ylabel("Survival Estimate")
plt.title("Survival Estimate with Confidence Interval")
plt.legend()
plt.show()
