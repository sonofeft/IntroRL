import os, sys
import glob
import importlib
from introrl.mdp_data import simple_grid_world

r"""
This file builds all the ".mdp_pickle" files for mdp_data environments in IntroRL
It places them in the users HOME directory:
  e.g. on Windows = C:\Users\<your name>\IntroRL_MDP
       on Linux = /home/<your_name>
"""

def main():
    USER_HOME_DIR = os.path.dirname( os.path.expanduser('~/') )
    FULL_DEST_DIR = os.path.join( USER_HOME_DIR, 'IntroRL_MDP' )
    MDP_SRC_PATH  = os.path.dirname( simple_grid_world.__file__ )

    print('        USER_HOME_DIR:',USER_HOME_DIR)
    print('IntroRL MDP Directory:', FULL_DEST_DIR)
    print('         MDP_SRC_PATH:',MDP_SRC_PATH)

    if not os.path.isdir( FULL_DEST_DIR ):
        print('Making Dir:', FULL_DEST_DIR)
        os.mkdir( FULL_DEST_DIR )

    mdp_pyL = glob.glob( os.path.join( MDP_SRC_PATH, '*.py') )
    mdp_pyL = reversed( mdp_pyL ) # make 1st ones go a little faster.

    for fname in mdp_pyL:
        src_code = open(fname, 'r').read()
        name_loc = src_code.find('if __name__ == "__main__"')
        if name_loc>=0:
            lineL = src_code[name_loc:].split('\n')
            call_src = ''
            for line in lineL[1:]:
                if line.find('=')>0:
                    call_src = line.split('=')[-1].strip()
                    break
                
            if call_src.endswith('()'):
                module_name =  os.path.split( fname )[-1][:-3]
                
                print('GET:',fname)
                print('    ',module_name,call_src)
                
                mdp_module = importlib.import_module( 'introrl.mdp_data.' + module_name)
                
                #print('    ',mdp_module)
                if hasattr(mdp_module, call_src[:-2]):
                    env = getattr(mdp_module, call_src[:-2])() # call factory routine
                    nick_name = env.make_pickle_filename( env.name )[:-11]
                    print('     SAVING:',nick_name)
                    
                    dest_fname = os.path.join( FULL_DEST_DIR, nick_name )
                    env.save_to_pickle_file( fname=dest_fname )
                    
            else:
                print('IGNORE:',fname)

if __name__=="__main__":
    
    main()
