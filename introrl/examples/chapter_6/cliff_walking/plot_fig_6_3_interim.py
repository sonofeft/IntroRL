import matplotlib.pyplot as plt

from build_fig_6_3_interim_data import dataD
from build_fig_6_3_interim_data import  EPSILON, ALPHA_LIST, RUN_COUNT

ExpSarsa_raveD = dataD['ExpSarsa_raveD']
Qlearn_raveD = dataD['Qlearn_raveD']
Sarsa_raveD = dataD['Sarsa_raveD']

interim_esL = []
interim_sL  = []
interim_qL  = []


qalphaL = [0.0978583,0.149688,0.198927,0.248165,0.298268,0.34837,0.399336,0.448575,0.497813,0.548779,0.598018,0.648984,0.698223,0.749189,0.797564,0.84853,0.896904,0.947871,0.996245]
qrsumL  = [-149.742,-123.844,-108.85,-98.9193,-92.6881,-88.5989,-84.8992,-82.5625,-80.6153,-78.8628,-77.305,-76.1367,-75.3578,-74.3841,-73.6053,-73.4105,-73.0211,-73.2158,-74.5789]

salphaL = [0.10045,0.149688,0.201185,0.250156,0.300896,0.350457,0.400018,0.450759,0.49973,0.550471,0.599441,0.649592,0.699743,0.749894,0.799455,0.850196,0.899757,0.949317,1.00006]
srsumL  = [-139.227,-113.913,-99.4746,-90.1648,-83.6478,-78.7269,-75.4019,-73.0079,-71.279,-70.215,-69.55,-68.885,-69.018,-68.885,-68.752,-68.885,-68.752,-68.885,-72.077]

esalphaL = [0.099869,0.14943,0.199581,0.249732,0.299882,0.349443,0.399594,0.450335,0.498126,0.549457,0.599607,0.649168,0.698729,0.74947,0.799621,0.849182,0.898153,0.950073,0.998454]
esrsumL  = [-133.732,-107.797,-91.5712,-80.7983,-72.8184,-67.2324,-62.5775,-59.2525,-56.3266,-54.0656,-52.3366,-50.7406,-49.1446,-48.0806,-47.5487,-46.2187,-45.6867,-44.7557,-44.3567]


for alpha in ALPHA_LIST:
    interim_esL.append( ExpSarsa_raveD[alpha][0].get_ave() )
    interim_sL.append( Sarsa_raveD[alpha][0].get_ave() )
    interim_qL.append( Qlearn_raveD[alpha][0].get_ave() )


fig, ax = plt.subplots()
plt.title('Exp-Sarsa, Sarsa, Q-Learning Cliff Walking\n'+\
          'Epsilon=%g (Interim has 100 episodes averaged over %i runs)\n'%(EPSILON, RUN_COUNT) )

fig.subplots_adjust(top=0.8)

plt.xlabel('Learning Rate (alpha)')
plt.ylabel('Reward Sum per Episode')
        
plt.plot(ALPHA_LIST, interim_esL, 'rx-', label='Exp-Sarsa IntroRL' )
plt.plot(ALPHA_LIST, interim_sL,  'bv-', label='Sarsa IntroRL' )
plt.plot(ALPHA_LIST, interim_qL,  'ks-', label='Q-learning IntroRL' )
        
plt.plot(esalphaL, esrsumL, 'r:', label='Exp-Sarsa Sutton' )
plt.plot(salphaL, srsumL,  'b:', label='Sarsa Sutton' )
plt.plot(qalphaL, qrsumL,  'k:', label='Q-learning Sutton' )

plt.legend()

#plt.ylim(bottom=-120)
plt.grid()

fig.savefig("figure_6_3_cliff_walking_interim.png")

plt.show()


