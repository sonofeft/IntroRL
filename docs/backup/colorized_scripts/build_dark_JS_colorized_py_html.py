import os
import glob
import html

here = os.path.abspath(os.path.dirname(__file__))
up_one = os.path.split( here )[0]
code_dir = os.path.join( up_one, 'introrl' )
dest_dir = os.path.join( here, '_static', 'colorized_scripts')

dpL = glob.glob( os.path.join( code_dir, 'dp_funcs', '*.py') )

def make_html( fpath, script_name ):
    print('Making:', script_name)
    fInp = open(fpath, 'r')
    code = html.escape( fInp.read(), quote=True )
    fInp.close()
    
    html_src = TEMPLATE1.format(py_file_name=script_name) + code + TEMPLATE2
    
    html_fname = script_name[:-2] + 'html'
    fOut = open( os.path.join( dest_dir, html_fname ), 'w' )
    fOut.write( html_src )
    fOut.close()

def build_all():
    
    for fpath in dpL:
        script_name = os.path.split( fpath )[-1]
        if not script_name.startswith('__'):
            make_html(fpath, script_name )
    



TEMPLATE1 = """<!DOCTYPE html>
<meta charset=utf-8>

<HEAD> 
<title>{py_file_name}</title>
<link href="rb_blackboard.css" rel="stylesheet" type="text/css" media="screen">
</HEAD> 


<body>

<pre>
<code data-language="python">"""

TEMPLATE2 = """</code>
</pre>

    <script src="rainbow-custom.min.js"></script>
    <script src="rainbow.linenumbers.min.js"></script>

</body>"""

if __name__=="__main__":
    
    build_all()