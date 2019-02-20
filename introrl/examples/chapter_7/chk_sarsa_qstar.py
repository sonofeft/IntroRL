import sys

from introrl.agent_supt.nstep_sarsa_qstar_walker import NStepSarsaQStarFinder

from introrl.policy import Policy
from introrl.agent_supt.action_value_coll import ActionValueColl
from introrl.agent_supt.nstep_sarsa_eval_walker import NStepSarsaWalker
from introrl.mdp_data.random_walk_generic_mrp import get_random_walk

DO_QSTAR = False

ALPHA = 0.2
GAMMA=0.9
NSTEPS = 8

rw_mrp = get_random_walk(Nside_states=9, win_reward=1.0, lose_reward=-1.0, step_reward=0.0)

if DO_QSTAR:
    EPSILON = 0.1
    walker = NStepSarsaQStarFinder(rw_mrp, Nsteps=NSTEPS, epsilon=EPSILON)
    av_coll = walker.av_coll
else:
    policy = Policy( environment=rw_mrp )
    walker = NStepSarsaWalker(rw_mrp, Nsteps=NSTEPS, policy=policy)
    av_coll = ActionValueColl( rw_mrp, init_val=0.0 )

#walker.av_coll.summ_print( fmt_Q='%.3f', none_str='*', show_states=True, show_last_change=True, show_policy=True)
print('<>'*60)

for _ in range(200):
    if DO_QSTAR:
        walker.do_sarsa_action_value_updates( alpha=ALPHA, gamma=GAMMA,start_state_hash='C')
    else:
        walker.do_sarsa_action_value_updates(av_coll, alpha=ALPHA, gamma=GAMMA,start_state_hash='C')

av_coll.summ_print( fmt_Q='%.4f', none_str='*', show_states=True, show_last_change=True, show_policy=True)
print('<>'*60)

sv = av_coll.build_sv_from_av()
sv.summ_print()