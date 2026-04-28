import numpy as np
import json

def compute_msd(positions, max_lag=429):
    msd = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        displacements = positions[lag:] - positions[:-lag]
        squared_displacements = np.sum(displacements ** 2, axis=1)
        msd[lag - 1] = np.mean(squared_displacements)
    return msd

with open("../auto_agents_1000_all_data.json", 'r') as f:
    data = json.load(f)
import matplotlib.pyplot as plt

positions = data["positions"]
msds = []
slopes = []
plt.figure(figsize=(8, 6))

for agent in range(len(positions)):
    agent_positions = np.array(positions[agent])
    msd = compute_msd(agent_positions, max_lag=len(agent_positions)-1)
    msds.append(msd)
    lags = np.arange(1, len(msd) + 1)
    #print log-log slope of the mean msd
    log_lags = np.log(lags)
    log_msd = np.log(msd)
    slope, intercept = np.polyfit(log_lags, log_msd, 1)
    slopes.append(slope)
    plt.plot(lags, msd, label=f'agent {agent}')



plt.xlabel('Lag Time')
plt.ylabel('Mean Squared Displacement')
plt.title('Mean Squared Displacement of Agents')
plt.legend()
plt.grid()
plt.show()


mean_slope= np.mean(slopes)
print("Mean Log-log slope of MSD:", mean_slope)

mean_msd = np.mean(np.array(msds), axis=0)
lags = np.arange(1, len(mean_msd)+1)

plt.figure(figsize=(8, 6))
plt.plot(lags, mean_msd, label='Mean MSD')
plt.xlabel('Lag Time')
plt.ylabel('Mean Squared Displacement')
plt.title('Mean Squared Displacement of Agents')
plt.legend()
plt.grid()
plt.show()

