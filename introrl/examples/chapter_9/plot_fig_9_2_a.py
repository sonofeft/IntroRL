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

line1 = ax.plot( stateL, true_valueL,  'r-', label='True Value, v(pi)' )

w_vector = [-0.65955679, -0.4312261 , -0.25431259, -0.1038249 ,  0.02766936,
            0.14877805,  0.28662611,  0.42889903,  0.58372318,  0.76130435]
mc_xL = []
mc_yL = []
x = 1
for w in w_vector:
    mc_xL.append( x )
    mc_yL.append( w )
    mc_xL.append( x+99 )
    mc_yL.append( w )
    x += 100
line2 = ax.plot( mc_xL, mc_yL,  'g-', label='Approx TD(0), vhat' )


ax.legend()
ax.grid( False )

fig.tight_layout()
fig.savefig("figure_9_2_a.png")

plt.show()

