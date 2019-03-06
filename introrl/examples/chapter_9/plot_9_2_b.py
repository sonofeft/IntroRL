import matplotlib.pyplot as plt
from numpy import array
from introrl.agent_supt.learning_tracker import LearnTracker
from introrl.policy import Policy
from introrl.agent_supt.epsilon_calc import EpsilonGreedy
from introrl.agent_supt.alpha_calc import Alpha
from introrl.agent_supt.episode_maker import make_episode
from introrl.agent_supt.nstep_td_eval_walker import NStepTDWalker
from introrl.black_box_sims.random_walk_1000 import RandomWalk_1000Simulation
from introrl.utils.running_ave import RunningAve

from rw1000_func import AggregatedRandomWalk1000_Func

GAMMA=1.0
AVE_OVER = 100

RW = RandomWalk_1000Simulation()

policy = Policy(environment=RW)
policy.intialize_policy_to_equiprobable( env=RW )

episode = make_episode(500, policy, RW, max_steps=10000)

# ----------------------------------------- generate data -------------
alphaL = [0.05*n for n in range(21)]

nstepL =  [1,2,4,8, 16, 32, 64]

nstep_walkerL = []
ave_rms_aveD = {} # index=(alpha, Nsteps), value=RunningAve
sv_funcD = {} # index=(alpha, Nsteps), value=AggregatedRandomWalk1000_Func
    

# create data structures
for Nsteps in nstepL:
    nstep_walkerL.append( NStepTDWalker(RW, Nsteps=Nsteps,  episode_obj=episode) )
    for alpha in alphaL:
        ave_rms_aveD[ (alpha, Nsteps) ] = RunningAve()
        sv_funcD[ (alpha, Nsteps) ] = AggregatedRandomWalk1000_Func( RW, num_regions=20 )

# begin main loop over runs
for loop in range(AVE_OVER): # average rms curves over AVE_OVER runs
    if loop%10==0:
        print(loop, end='')
    else:
        print('.', end='')
    
    # set state w_vector variables to 0.0
    for Nsteps in  nstepL :
        for  alpha in alphaL:
            sv_funcD[ (alpha, Nsteps) ].init_w_vector()

    # look at first 10 episodes
    for _ in range(10):
        episode = make_episode(500, policy, RW, max_steps=10000)
        
        # have each alpha and Nsteps use the same episode_obj
        for istep,Nsteps in enumerate( nstepL ):
            nstep_walkerL[ istep ].set_episode_obj( episode )
            
            for alpha in alphaL:
                nstep_walkerL[ istep ].do_td_sv_function_updates( sv_funcD[ (alpha, Nsteps) ], alpha=alpha, gamma=GAMMA)

                # get the RMS after each episode.
                rms = sv_funcD[ (alpha, Nsteps) ].calc_rms_error()
                ave_rms_aveD[ (alpha, Nsteps) ].add_val( rms )
        
print()

colorD = {1:'r', 2:'g', 4:'b', 8:'k', 16:'m', 32:'c', 64:'y'}
fig, ax = plt.subplots( figsize=(8,6) )

for Nsteps in  nstepL :
    ave_rmsL = []
    for  alpha in alphaL:
        ave_rmsL.append( ave_rms_aveD[ (alpha, Nsteps) ].get_ave() )
    ax.plot(alphaL, ave_rmsL, '%s-'%colorD[Nsteps], label='Nsteps=%i, IntroRL'%Nsteps )

plt.ylim([0.25, 0.55])
    
plt.ylabel('Ave. RMS Error (%i experiments)'%AVE_OVER)
plt.xlabel('Alpha ' + r'($\alpha$)')

# Zhang Results
alphas = array([0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 ,
       0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95, 1.  ])
errors_1 = array([0.54346485, 0.52301194, 0.50422126, 0.48708255, 0.47056158,
       0.45567889, 0.44140045, 0.42716099, 0.41436938, 0.40267275,
       0.39003853, 0.38330027, 0.37146481, 0.36243917, 0.35870504,
       0.35365013, 0.35120526, 0.35458384, 0.3644122 , 0.38543964,
       0.42865446])
errors_2 = array([0.54346485, 0.50518009, 0.47220401, 0.44207045, 0.41447032,
       0.39015081, 0.37025748, 0.35328462, 0.33449132, 0.32323272,
       0.30557266, 0.31316945, 0.30634503, 0.30789654, 0.31302827,
       0.31636149, 0.32863725, 0.34385915, 0.36679105, 0.39808092,
       0.46138648])
errors_4 = array([0.54346485, 0.47593027, 0.42042463, 0.37794216, 0.33931033,
       0.31800676, 0.29180674, 0.28936826, 0.28595051, 0.29069073,
       0.28938636, 0.30235311, 0.32006741, 0.3328899 , 0.34431006,
       0.36134708, 0.38558591, 0.4047511 , 0.43499991, 0.47766768,
       0.51614276])
errors_8 = array([0.54346485, 0.4341758 , 0.36464929, 0.32592566, 0.29949156,
       0.29720932, 0.30412968, 0.3203433 , 0.33691732, 0.34452017,
       0.3620062 , 0.37790436, 0.39290479, 0.41713231, 0.43736459,
       0.46064813, 0.47416269, 0.4973127 , 0.52604507, 0.55826309,
       0.60975813])
errors_16 = array([0.54346485, 0.39700918, 0.34108609, 0.32901849, 0.34715719,
       0.35510326, 0.37792898, 0.40125412, 0.42833899, 0.44357229,
       0.46547058, 0.4831478 , 0.50590963, 0.52734658, 0.5353047 ,
       0.55963969, 0.57939287, 0.60298622, 0.62398708, 0.63582318,
       0.67225318])
errors_32 = array([0.54346485, 0.39239582, 0.37649445, 0.40578772, 0.43884392,
       0.46203138, 0.49513178, 0.5169309 , 0.54472173, 0.56154064,
       0.5883069 , 0.59643114, 0.62920153, 0.63582551, 0.64777206,
       0.66612973, 0.67598703, 0.68303723, 0.7102483 , 0.71351987,
       0.73641854])
errors_64 = array([0.54346485, 0.43187701, 0.46706372, 0.49905741, 0.55088214,
       0.55765315, 0.60665472, 0.63412362, 0.63723193, 0.67687787,
       0.66504584, 0.68964987, 0.70802699, 0.73256161, 0.72771667,
       0.74394468, 0.74957095, 0.75065964, 0.76162076, 0.76997655,
       0.7911325 ])
zhang_rmsD = {1:errors_1, 2:errors_2, 4:errors_4, 8:errors_8, 16:errors_16, 32:errors_32, 64:errors_64}

for Nsteps in  nstepL :
    ax.plot(alphas, zhang_rmsD[Nsteps], '%s:'%colorD[Nsteps], label='Nsteps=%i, Zhang'%Nsteps )


#ax.legend( fontsize=8, loc='upper left', bbox_to_anchor=(0.15, 0.99), framealpha=0.25 )
ax.legend( fontsize=8, loc='best', framealpha=0.25 )

ax.set(title='Figure 9.2 Gamma = %g, Ave Over %i Runs\n'%(GAMMA, AVE_OVER) +\
             'Aggregated Random Walk 1000 with Equiprobable Policy' )
fig.savefig("figure_9_2_random_walk_nstep.png")
plt.show()
