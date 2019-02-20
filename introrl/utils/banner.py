#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

        
def floatDammit( val=0.0 ):
    val = str( val )
    v = val.strip()
    if v.lower() in ['nan','inf','-inf']:
        if v.lower() == '-inf':
            v = -1.0E99
        elif v.lower() == 'inf':
            v = 1.0E99
        else:
            v = 0.0
    try:
        v = float(v)
    except:
        v = 0.0
    return v

def is_float( fval ):
    if type(fval)==type(11.11):
        return 1
    if type(fval)==type('string'):
        if fval.find('.')==-1:
            return 0
    try:
        if float(fval) > -float("inf"):
            return 1
    except:
        #print 'Failed is_float',fval,type(fval)
        pass
    return 0

def get_banner_chars( banner_char='' ):
    
    if banner_char:
        ul = ur = ll =  lr = vr = vl = hb = ht = vert = hor = um = lm = banner_char
        if banner_char == '=':
            vl = vr = vert = '|'
            ul = '+'; ur = '+'; ll = '+'; lr = '+'
    else:# printable characters
        ul = '+'; ur = '+'; ll = '+'; lr = '+'; vert = '|'; hor = '-'
        vl = vr = vert
        ht = hb = hor
        um = lm = '+'
        
    return ul, ur, ll,  lr, vr, vl, hb, ht, vert, hor, um, lm


def banner(s, banner_char='', leftMargin=0, just='center'):
    
    ul, ur, ll,  lr, vr, vl, hb, ht, vert, hor, um, lm = get_banner_chars( banner_char )
    
    sL = s.split('\n')
    L = 0
    for s in sL:
        s = s.strip()
        ss = '  ' + s + '  '
        L = max(L, len(ss))
    
    top = leftMargin*' ' + ul + L*ht + ur
    bot = leftMargin*' ' + ll + L*hb + lr
    
    midL = []
    for s in sL:
        s = s.strip()
        if just=='center':
            midL.append( leftMargin*' ' + vl + '%s'%(s.center(L),) + vr )
        elif just=='right':
            midL.append( leftMargin*' ' + vl + '%s'%(s.rjust(L),) + vr )
        else:
            midL.append( leftMargin*' ' + vl + '%s'%(s.ljust(L),) + vr )
    
    pad = ''
    if just=='center':
        Lpad = (80 - len(top)) // 2
        pad = ' '*Lpad
    elif just=='right':
        Lpad = 79 - len(top)
        pad = ' '*Lpad

    
    print( pad+top )
    for mid in midL:
        print( pad+mid )
    print( pad+bot )

def getStr( val ):
    if is_float(val):
        val = floatDammit( val )
        #return ('%f' % val).rstrip('0').rstrip('.')
        return '%g'%val 
    else:
        return str( val )

def show_table( titleL=None, LOL=None, banner_char='', header_sep=True ):
    '''Number of columns = len(titleL)
       Lists of values in list of lists LOL
    '''
    #ul = chr(218); ur = chr(191); ll = chr(192); lr = chr(217); vert = chr(179); hor = chr(196)
    #um = chr(194); lm = chr(193)
    #vl = vr = vert
    #ht = hb = hor
    ul, ur, ll,  lr, vr, vl, hb, ht, vert, hor, um, lm = get_banner_chars( banner_char )
    
    
    Ncol = len(titleL)
    lenL = [] # allocation length for each column 
    for i,L in enumerate(LOL):
        for j in range( Ncol ):
            lenL.append( len(titleL[j]) )
            for val in L:
                lstr = len( getStr(val) )
                if lstr > lenL[-1]:
                    lenL[-1] = lstr
    #print 'lenL =',lenL
    # print top line 
    sL = [ul]
    for i in range(Ncol):
        sL.append( hor*(lenL[i]+2) )
        if i < Ncol-1:
            sL.append( um )
    sL.append(ur)
    print( ''.join(sL) )
    
    # print titles
    sL = []
    for i,title in enumerate(titleL):
        sL.append( title.center( lenL[i]+2 ) )
    print( vert + vert.join(sL) + vert )

    if header_sep:
        # print header separation line
        sL = [ul]
        for i in range(Ncol):
            sL.append( hor*(lenL[i]+2) )
            if i < Ncol-1:
                sL.append( um )
        sL.append(ur)
        print( ''.join(sL) )

    # print content
    for i in range( len(LOL) ):
        sL = []
        for j in range( Ncol ):
            sL.append( getStr(LOL[i][j]).center( lenL[j]+2 ) )
        print( vert + vert.join(sL) + vert )

    # print bottom line 
    sL = [ll]
    for i in range(Ncol):
        sL.append( hor*(lenL[i]+2) )
        if i < Ncol-1:
            sL.append( lm )
    sL.append(lr)
    print( ''.join(sL) )
    

if __name__=="__main__":
    
        
    
    print()
    # --------------
    
    for banner_char in ('','.','*',':','='):
        banner('Simple Banner char="%s"'%banner_char, banner_char=banner_char)
        print()
        show_table( ['row1 banner_char="%s"'%banner_char,'second Row'], 
                    [[1,2,3.333],['a','seven',123.4]], 
                    banner_char=banner_char )
        print()
    
    