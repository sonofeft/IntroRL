#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

def is_literal_str( s ):
    """check if s is a string in double quotes."""
    if type(s) == type('string'):
        if s.startswith('"') and s.endswith('"'):
            return True
        else:
            return False
    else:
        return False
    

def print_string_rows( rows_outL, row_tickL=None, col_tickL=None, const_col_w=True, 
                       line_chr='=', left_pad='',
                       header='',  x_axis_label='', y_axis_label='' ,
                       justify=''): # left, right, center
    """
    print as a grid, the rows of strings in "rows_outL
    If row_tickL is input, each row is followed by that text
    the max length of any string in each column is given by lmaxL.
    
    justify can be left, right, center for each entry.
    
    If input, x_axis_label and y_axis_label add labels to the x and y axes
    """
    # ---------------------------------------------------------------------------------
    # look through rows_outL for any carriage returns, if found, make multiline output
    do_multiline = False
    num_lines_in_rowL = [] # number of lines in each row
    for row in rows_outL:
        num_lines = 1
        for s in row:
            num_lines = max(num_lines, s.count('\n') + 1)
            if num_lines > 1:
                do_multiline = True
        num_lines_in_rowL.append( num_lines )
                
    if do_multiline:
        new_rows_outL = []
        if row_tickL is not None:
            new_row_tickL = []
            
        for irow, row in enumerate( rows_outL ):
            num_lines = num_lines_in_rowL[ irow ]
            
            for i in range( num_lines ):
                new_row = []
                for s in row:
                    sL = str(s).split('\n')
                    if i < len(sL):
                        new_row.append( sL[i] )
                    else:
                        new_row.append( '' )
                new_rows_outL.append( new_row )
                if row_tickL is not None:
                    if i>0:
                        new_row_tickL.append( '..'+str(row_tickL[irow]) )
                    else:
                        new_row_tickL.append( str(row_tickL[irow]) )
                    
        rows_outL = new_rows_outL
        if row_tickL is not None:
            row_tickL = new_row_tickL
        
    
    # ---------------------------------------------------------------------------------
    # figure out lengths of tick labels and make sure they are strings
    if row_tickL is None:
        row_tickL = []
        row_tick_len = 0
    else:
        row_tickL = [str(v) for v in row_tickL]
        row_tick_len = max( [len(v) for v in row_tickL] )
        
    if col_tickL is None:
        col_tickL = []
        col_tick_len = 0
    else:
        col_tickL = [str(v) for v in col_tickL]
        col_tick_len = max( [len(v) for v in col_tickL] )
        
    # calc number of rows and columns in data 
    Nrows = len(rows_outL)
    Ncols = max( [len(row) for row in rows_outL] )
    
    # use Nrows to figure out if y_axis_label will fit as one char per line or more
    y_axis_charL = []
    ylab_len = len(y_axis_label)
    
    if Nrows >= ylab_len: # one letter per row, centered if necessary
        jstart = int( (Nrows - ylab_len)//2 )
        if jstart:
            y_axis_charL.extend( [' ']*jstart )
        for j in range( ylab_len ):
            y_axis_charL.append( y_axis_label[j] )
        while len(y_axis_charL) < Nrows:
            y_axis_charL.append( ' ' )
    else:
        #c_per_row, extra =  divmod(ylab_len, Nrows)
        d_let = float(ylab_len) / float(Nrows)
        i = 0
        for n in range(1, Nrows+1):
            iend = int( n * d_let)
            y_axis_charL.append( y_axis_label[i:iend] )
            i = iend
            if n>0 and i>0 and y_axis_charL[-1].endswith(' '):
                i = i - 1  # try not to lose spaces in 
        
        while len(y_axis_charL) < Nrows:
            y_axis_charL.append( ' ' )
    
    
    # get the width of each column (based on largest entry)
    lmaxL = [1]*Ncols # track longest value shown
    for i in range( Nrows ):
        rowL = rows_outL[ i ]
        for j in range( Ncols ):
            if j < len(rowL):
                s = rowL[j]
                lmaxL[j] = max( lmaxL[j], len(s), col_tick_len )
                
    # adjust column widths if they should all be the same
    if const_col_w:
        lmax = max( lmaxL )
        lmax = max(lmax, col_tick_len)
        lmaxL = [lmax] * Ncols
    
    # total character width including spacing character between columns.
    ltot = sum( lmaxL ) + len(lmaxL)-1

    # add bracketing spaces to header and x_axis_label
    if header:
        header_str = ' ' + header.strip() + ' '
    else:
        header_str = ''

    if x_axis_label:
        footer_str = ' ' + x_axis_label.strip() + ' '
    else:
        footer_str = ''

    # figure out any left padding
    l_hdft = max(len(header_str)+6, len(footer_str)+6)
    l_pad = max(0, int( (l_hdft - ltot) / 2))

    if l_hdft > ltot:
        print(left_pad, header_str.center(l_hdft, line_chr) )
    else:
        print(left_pad,  header_str.center(ltot, line_chr) )
    
    # print the rows with data and y_axis_label
    for i,L in enumerate(rows_outL):
        if l_pad:
            print(left_pad,  ' '*l_pad , end='')
        else:
            print(left_pad , end='')
        
        for j,s in enumerate(L):
            if justify == 'left':
                sout = s.ljust(lmaxL[j], ' ')
            elif justify == 'right':
                sout = s.rjust(lmaxL[j], ' ')
            elif justify == 'center':
                sout = s.center(lmaxL[j], ' ')
            else:
                fmt = '%' + '%is'%lmaxL[j]
                sout = fmt%s
            
            print( sout, end=' ')
        
        # add row tick labels if input.
        if len(row_tickL) > i:
            sout = row_tickL[i].ljust(row_tick_len, ' ')
            print('| %s'%sout, end='')
        
        # add y_axis_label if input
        if y_axis_charL:
            print( ' || %s'%y_axis_charL[i], end='' )
        print()
    
    # add column tick labels
    if col_tickL:
        if l_pad:
            print(left_pad,  ' '*l_pad , end='')
            l_col_tick = max(l_hdft, ltot) - l_pad
        else:
            print(left_pad , end='')
            l_col_tick = max(l_hdft, ltot)
            
        out_strL = [ s.center(lmaxL[j], '_') for j,s in enumerate(col_tickL) ]
        print( '_'.join( out_strL ).ljust(l_col_tick, '_') )
    
    if x_axis_label:
        print(left_pad,  footer_str.center(max(l_hdft, ltot), line_chr) )
    
    #print(line_chr*ltot)
    
    return lmaxL # return the number of characters in each row.


if __name__ == "__main__": # pragma: no cover
    
    rows_outL = [['A','B\nCD','E'], ['FGH','I','JK\nLMN\nOP']]
    print_string_rows( rows_outL, 
                       row_tickL=['1st','2nd'], 
                       col_tickL=['eeeny','meeny','mieny'],
                       const_col_w=True,
                       header='Partial Alphabet', 
                       x_axis_label='Alphabet Stuff', y_axis_label='sideview')
                           
                           
                           