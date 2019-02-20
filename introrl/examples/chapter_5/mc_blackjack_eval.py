
from introrl.mc_funcs.mc_fv_prediction import mc_first_visit_prediction
from introrl.black_box_sims.blackjack_sim import BlackJackSimulation
from introrl.policy import Policy
from introrl.agent_supt.state_value_run_ave_coll import StateValueRunAveColl

BJ = BlackJackSimulation()

pi = Policy(  environment=BJ  )

# default policy is hit on everything except 20 & 21.
pi.set_policy_from_piD( BJ.get_default_policy_desc_dict() )

sv = StateValueRunAveColl( BJ )

if 1:
    mc_first_visit_prediction( pi, sv, max_num_episodes=10000, max_abserr=0.001, gamma=1.0)
    sv.save_to_pickle_file( fname='mc_blackjack_10000_eval')
else:
    mc_first_visit_prediction( pi, sv, max_num_episodes=500000, max_abserr=0.001, gamma=1.0)
    sv.save_to_pickle_file( fname='mc_blackjack_500000_eval')
    