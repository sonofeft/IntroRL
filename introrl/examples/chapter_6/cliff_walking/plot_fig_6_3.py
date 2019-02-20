import matplotlib.pyplot as plt

from build_fig_6_3_data import dataD
from build_fig_6_3_data import  EPSILON, ALPHA_LIST, RUN_COUNT

ExpSarsa_raveD = dataD['ExpSarsa_raveD']
Qlearn_raveD = dataD['Qlearn_raveD']
Sarsa_raveD = dataD['Sarsa_raveD']

interim_esL = []
interim_sL  = []
interim_qL  = []
interim_tdL  = []

asym_esL = []
asym_sL  = []
asym_qL  = []
asym_tdL  = []

for alpha in ALPHA_LIST:
    interim_esL.append( ExpSarsa_raveD[alpha][0].get_ave() )
    interim_sL.append( Sarsa_raveD[alpha][0].get_ave() )
    interim_qL.append( Qlearn_raveD[alpha][0].get_ave() )
    #interim_tdL.append( TD0_raveD[alpha][0].get_ave() )

    asym_esL.append( ExpSarsa_raveD[alpha][1].get_ave() )
    asym_sL.append( Sarsa_raveD[alpha][1].get_ave() )
    asym_qL.append( Qlearn_raveD[alpha][1].get_ave() )
    #asym_tdL.append( TD0_raveD[alpha][1].get_ave() )


fig, ax = plt.subplots()
plt.title('Exp-Sarsa, Sarsa, Q-Learning Cliff Walking\n'+\
          'Epsilon=%g (Interim has 100 episodes averaged over %i runs)\n'%(EPSILON, RUN_COUNT) +\
          '(Asymtotic has 1,000 episodes averaged over %i runs)\n'%( RUN_COUNT) )

fig.subplots_adjust(top=0.8)

plt.xlabel('Learning Rate (alpha)')
plt.ylabel('Reward Sum per Episode')
    
plt.plot(ALPHA_LIST, asym_esL, 'rx-', label='Exp-Sarsa-asym' )
plt.plot(ALPHA_LIST, asym_sL,  'bv-', label='Sarsa-asym' )
plt.plot(ALPHA_LIST, asym_qL,  'ks-', label='Q-learning-asym' )
    
plt.plot(ALPHA_LIST, interim_esL, 'rx:', label='Exp-Sarsa-interim' )
plt.plot(ALPHA_LIST, interim_sL,  'bv:', label='Sarsa-interim' )
plt.plot(ALPHA_LIST, interim_qL,  'ks:', label='Q-learning-interim' )


plt.legend()

plt.ylim(bottom=-140, top=0)
plt.xlim(left=0.1)
plt.grid()

fig.savefig("figure_6_3_cliff_walking_plot.png")

plt.show()


