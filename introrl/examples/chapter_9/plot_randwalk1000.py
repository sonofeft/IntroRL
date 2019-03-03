import matplotlib.pyplot as plt

from introrl.utils import pickle_esp

env, state_values, policy = pickle_esp.read_pickle_file( fname='random_walk_1000_PI_eval')

#state_values.summ_print()

stateL = list( range(2,1000) )
true_valueL = [ state_values.VsD[i] for i in stateL]


fig, ax = plt.subplots()

plt.title('Random Walk 1000')

plt.xlabel('State')
plt.ylabel('Value Scale')
plt.grid()

plt.plot( stateL, true_valueL,  'r-', label='True Value, v(pi)' )

plt.legend()

fig.savefig("figure_9_1.png")

plt.show()

