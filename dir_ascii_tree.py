import os
import subprocess
#subprocess.call(['tree', '/A','/F'], shell=True)
out = subprocess.check_output(['tree', '/A','/F'], shell=True)

here = os.path.abspath( os.getcwd() )
print( here )

outL = out.split('\n')
for i,line in enumerate(outL):
    line = line.strip()
    if line.endswith('.pyc'):
        pass
    elif line.endswith('__pycache__'):
        pass
    elif i <= 1:
        pass
    elif i==2:
        print( '|' )
    else:
        print(line)
