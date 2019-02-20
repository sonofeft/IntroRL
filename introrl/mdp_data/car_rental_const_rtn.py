from math import exp, factorial

from introrl.layouts.generic_layout import GenericLayout
from introrl.environments.env_baseline import EnvBaseline
from introrl.reward import Reward
from introrl.utils.functions import select_weighted_random # from list of (n, w) pairs

POISSON_UPPER_BOUND = 11 # taken from numpy implementation on web site.
MAX_CARS = 20
N1_LAMBDA = 3
N2_LAMBDA = 4

N1_RTNS = 3
N2_RTNS = 2

RENTAL_CREDIT = 10
MOVE_CAR_COST = 2

# make a quick look-up for poisson number
def poisson(n, lam):
    pnum = exp(-lam) * pow(lam, n) / factorial(n)
    return pnum


# make weighted lists of (n, prob) pairs for lam 2,3 and 4
# --------- lookup time is about 20x faster --------------
prob_lookupD = {} # index=(n,lam), value=prob

def fill_prob_lookupD(lam=3):
    for n in range(POISSON_UPPER_BOUND ):
        prob_lookupD[(n,lam)] = poisson(n, lam)    

for lam in [2,3,4]:
    fill_prob_lookupD( lam )


def get_prob_reward( s1, s2, a_desc):
    """
    Call add_results to save prob and rewards
    """

    for n1_rent_request in range( POISSON_UPPER_BOUND ):
        for n2_rent_request in range( POISSON_UPPER_BOUND ):
            
            n1 = int(min(s1 - a_desc, MAX_CARS))
            n2 = int(min(s2 + a_desc, MAX_CARS))
            
            actual_n1_rented = min(n1, n1_rent_request)
            actual_n2_rented = min(n2, n2_rent_request)
            
            reward = (actual_n1_rented + actual_n2_rented) * RENTAL_CREDIT \
                   - abs(a_desc) * MOVE_CAR_COST
                     
            prob_rented = prob_lookupD[ (n1_rent_request, N1_LAMBDA) ] \
                        * prob_lookupD[ (n2_rent_request, N2_LAMBDA) ]
            
            # next state
            sn1 = int(min(n1 - actual_n1_rented + N1_RTNS, MAX_CARS))
            sn2 = int(min(n2 - actual_n2_rented + N2_RTNS, MAX_CARS))
            
            add_results( s1, s2, a_desc, prob_rented, reward, sn1, sn2 )

# define all possible transitions.
total_probD = {} # index=(s1, s2, a_desc, sn_hash): value=t_prob total 
sum_prob_x_rewardD = {} # index=(s1, s2, a_desc, sn_hash): value=sum of t_prob * reward

def add_results( s1, s2, a_desc, prob_rented, reward, sn1, sn2 ):
    #print('s1=%2i, s2=%2i, a_desc=%2i, prob_rented%8.6f, reward%5.1f, sn1=%2i, sn2=%2i'%\
    #     (s1, s2, a_desc, prob_rented, reward, sn1, sn2))
         
    sn_hash = (sn1, sn2)
    key = (s1, s2, a_desc, sn_hash)
    
    if key not in total_probD:
        total_probD[key] = 0.0
        sum_prob_x_rewardD[key] = 0.0
        
    total_probD[key] += prob_rented # index=(s1, s2, a_desc, sn_hash): value=t_prob total 
    sum_prob_x_rewardD[key] += prob_rented * reward # index=(s1, s2, a_desc, sn_hash): value=sum of t_prob * reward
    

def get_env():

    env = EnvBaseline( name="Jacks Car Rental (const rtn)" ) # GenericLayout set below
    
    simplified_str ="""Shangtong Zhang's simplified model such that
the # of cars returned in daytime becomes constant
rather than a random value from poisson distribution, which will reduce calculation time
and leave the optimal policy/value state matrix almost the same"""    
    
    env.set_info( 'Example 4.2 from Sutton & Barto 2nd Edition page 81.\n' + simplified_str )

    
    # define all possible actions.
    saL = [] # a list of (s1, s2, adesc)
    s_hash_rowL = [] # layout rows for makeing 2D output

    
    for s1 in range( MAX_CARS + 1 ): # 20 cars max
        rowL = [] # row of s_hash_rowL
        
        for s2 in range( MAX_CARS + 1 ): # 20 cars max
            s_hash = (s1, s2)
            rowL.append( s_hash )
            
            for a_desc in range(-5, 6): # -5 moves 5 cars from 2nd to 1st. +5 from 1st to 2nd.
                
                if a_desc < 0: # can only move cars if they are present
                    if (abs(a_desc) <= s2):
                        env.add_action( s_hash, a_desc, a_prob=1.0 )
                        saL.append( (s1, s2, a_desc) )
                else:
                    if (a_desc <= s1): # can only move cars if they are present
                        env.add_action( s_hash, a_desc, a_prob=1.0 )
                        saL.append( (s1, s2, a_desc) )
        
        # use insert to put (0,0) at lower left
        s_hash_rowL.insert(0, rowL )# layout rows for makeing 2D output
    
    # ------------------------------
    # figure out transition probabilities and rewards
    for s1 in range( MAX_CARS + 1 ):
        for s2 in range( MAX_CARS + 1 ):
            for a_desc in range( -5, 6 ):
                get_prob_reward( s1, s2, a_desc)

    # ------------------------------                                                
        
    print('\nStarting to define car rental transitions')
    # with all the probability figured out, define all transitions
    for (s1, s2, a_desc, sn_hash), t_prob in total_probD.items():
        txr = sum_prob_x_rewardD[ (s1, s2, a_desc, sn_hash) ]
        rval = txr / t_prob
        env.add_transition( (s1,s2), a_desc, sn_hash, t_prob=t_prob, reward_obj=rval)
    
        #if s1==10 and s2==10:
        #    print('for (10,10) a_desc=',a_desc,' sn_hash=',sn_hash,'  t_prob=',t_prob,'  rval=',rval)
    
    print('Calling: env.define_env_states_actions')
    env.define_env_states_actions()  # send all states and actions to environment
    print('Environment Ready.')

    # If there is a start state, define it here.
    env.start_state_hash = (10,10)

    # define default policy (if any)
    env.default_policyD = {}


    # --------------------
    # define layout for output

    env.layout = GenericLayout( env, s_hash_rowL=s_hash_rowL, 
                                x_axis_label='#Cars at Second Location',
                                y_axis_label='#Cars at First Location')
    
    return env

if __name__ == "__main__": # pragma: no cover
    
    from introrl.dp_funcs.dp_value_iter import dp_value_iteration
    
    env = get_env()
    env.save_to_pickle_file( fname=None )
    #env.summ_print()
    
    if 0:
        policy, state_value = dp_value_iteration( env, do_summ_print=True, fmt_V='%.1f', fmt_R='%.1f',
                                                  max_iter=1000, err_delta=0.0001, 
                                                  gamma=0.9, iteration_prints=10)
                                              
    env.layout.s_hash_diagram( save_name='car_rental_const_rtn_diagram', none_str='*', do_show=True,
                               pad=0.05, scale=0.75, h_over_w=0.6)
    
    """  ===============   GIVES SAME RESULTS AS Shangtong Zhang Const Return SOLUTION   =================================    
    """
