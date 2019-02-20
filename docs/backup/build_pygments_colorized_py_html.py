import os
import glob
import html

from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

"""
Makes colorized HTML versions of *.py scripts
Places them in _static/colorized_scripts sub-directory of docs.
"""

here = os.path.abspath(os.path.dirname(__file__))
up_one = os.path.split( here )[0]
code_dir = os.path.join( up_one, 'introrl' )
dest_dir = os.path.join( here, '_static', 'colorized_scripts')

css_fname = os.path.join( dest_dir, 'pygments_default.css' )

lexer = get_lexer_by_name("python", stripall=False)
formatter = HtmlFormatter(linenos=True, cssclass="default")
css_src = open( css_fname ,'r').read()

# -------------- make lists of desired scripts here ------------------
py_fnameL = []
py_fnameL.extend( glob.glob( os.path.join( code_dir, 'dp_funcs', '*.py') ) )

def make_html( fpath, script_name ):
    print('Making:', script_name)
    
    pysrc_code = open(fpath, 'r').read()    
    colorized_py_html = highlight(pysrc_code, lexer, formatter)
    
    html_fname = script_name[:-2] + 'html'
    fOut = open( os.path.join( dest_dir, html_fname ), 'w' )
    
    fOut.write( "<html><head><title>%s</title></head><body>\n"%script_name )
    fOut.write( css_src )
    fOut.write( "\n<h2>%s</h2>\n"%script_name )
    fOut.write( colorized_py_html )
    fOut.write( "</body></html>" )
    
    fOut.close()


def build_all():
    
    for fpath in py_fnameL:
        script_name = os.path.split( fpath )[-1]
        if not script_name.startswith('__'):
            make_html(fpath, script_name )
    
if __name__=="__main__":
    
    build_all()