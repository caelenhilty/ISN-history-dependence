import numpy as np
import matplotlib.pyplot as plt
from sys import path
path.append(r"model")
import unified_model as model

simulateISP = model.simulateISP

thetaE = 2
thetaI = 5.5
WEE =  2.78415173362416
WEI =  4.826424331460146
WIE =  -0.6920758668123493
WII =  -0.8632121657324914

duration = 3
dt = 0.1e-3
IappI = np.zeros(int(duration/dt))
Iamp = 0.05
IappI[int(duration/(2*dt)):] = Iamp
rE, rI = simulateISP(dt, duration, 100, 10e-3, 10e-3, WEE, WEI, WIE, WII, thetaE, thetaI, IappI, np.zeros(int(duration/dt)),
                     rE0=5, rI0=10)

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
fig = plt.figure(layout = "constrained", figsize=(3,3), dpi =1200)
ax = plt.subplot(111)
plt.plot(np.arange(0, duration, dt)-1, rE, label='Excitatory', c= "#F58E89")
plt.plot(np.arange(0, duration, dt)-1, rI, label='Inhibitory', c='#7879ff')
plt.xlabel('Time (s)')
plt.ylabel('Firing Rate (Hz)')
ax.set(xlim=(0,2), ylim=(0,12))
ax.axvspan(0.5, 3, alpha=0.5, color='0.8')
ax.annotate(r"$I_{app}^{I}$ = " + str(Iamp), xy=(0.5, 0), xytext=(15*12, 15*12), textcoords='offset pixels', fontsize=14, color='0.2', fontfamily='serif')
plt.legend()
fig.savefig("figure1B_example_traces", dpi=1200)
# plt.show()