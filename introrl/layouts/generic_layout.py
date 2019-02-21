#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
    from matplotlib.font_manager import FontProperties

    got_matplotlob = True
except:
    got_matplotlob = False


from introrl.utils.grid_funcs import print_string_rows, is_literal_str

class GenericLayout( object ):
    
    def __init__(self, environment, s_hash_rowL=None, 
                 row_tickL=None, col_tickL=None,
                 x_axis_label='', y_axis_label='',
                 colorD=None, basic_color='', named_s_hashD=None):
        """
        Given an environment, a GenericLayout will help create 2D output.
    
        IF INPUT: s_hash_rowL is used to locate s_hash in output rows
            use None in s_hash_rowL to indicate empty grid locations.
            (e.g. s_hash_rowL=[['UL','UM','UR'],['LL',None,'LR']])
            
        If named_s_hashD is input, it is a dictionary with special names of s_hash
        values called out.
        """
        self.environment = environment
        self.s_hash_rowL = s_hash_rowL
        
        self.rev_lookupD = {} # index=s_hash, value=(row, col) in s_hash_rowL
        
        self.row_tickL = row_tickL
        self.col_tickL = col_tickL
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.colorD = colorD
        self.basic_color = basic_color
        
        if named_s_hashD is None:
            named_s_hashD = {}
        self.named_s_hashD = named_s_hashD
        
        # (x,y) are translated to the origin for any output plots
        self.xy_s_hashD = {} # index=s_hash: value=x,y location (with (0,0) at Lower Left)
        
        self.s_xyD = {}      # index=(x,y):  value=s_hash
        
        if s_hash_rowL is None:
            # output rows are not defined, so figure them out.
            xmax = 0
            ymax = 0
            misfit_xyL = [] # s_hash values that have no x,y
            for s_hash in environment.iter_all_states():

                #x = j
                #y = 0
                (x,y) = environment.get_layout_row_col_of_state( s_hash )
                #print('x,y=',x,y,' after get_layout_row_col_of_state for:',s_hash)
                
                if x is None:
                    misfit_xyL.append( s_hash ) # save unlocated s_hash for later
                else:
                    self.xy_s_hashD[s_hash] = (x,y) # index=s_hash: value=x,y location (with (0,0) at Lower Left)
                    
                    xmax = max(xmax, x)
                    ymax = max(ymax, y)
                    
                    self.s_xyD[(x,y)] =  s_hash  # index=(x,y):  value=s_hash
            # need to build self.s_hash_rowL
            self.s_hash_rowL = [] # will build s_hash_rowL
            for x in range( xmax+1 ):
                rowL = []
                for y in range( ymax+1 ):
                    rowL.append( self.s_xyD.get( (x,y), None ) )
                self.s_hash_rowL.append( rowL )
                
            self.height = len( self.s_hash_rowL )
            self.width = len(self.s_hash_rowL[0])
            
            if misfit_xyL:
                rowL = []
                x = self.height
                for ierr, s_hash in enumerate( misfit_xyL ):
                    y = ierr % self.width
                    self.xy_s_hashD[s_hash] = (x,y) # index=s_hash: value=x,y location (with (0,0) at Lower Left)
                    self.s_xyD[(x,y)] =  s_hash  # index=(x,y):  value=s_hash
                    rowL.append( s_hash )
                    
                    if (ierr>0) and (ierr%self.width==0):
                        self.s_hash_rowL.append( rowL )
                        rowL = []
                        x += 1
                if rowL:
                    self.s_hash_rowL.append( rowL )
                self.height = len( self.s_hash_rowL )
            #print('self.s_hash_rowL',self.s_hash_rowL)
            
        else:
            self.height = len( s_hash_rowL )
            self.width = 1
            for j,row in enumerate(s_hash_rowL):
                self.width = max( self.width, len(row) )
                
                y = self.height - j - 1
                for x, s_hash in enumerate( row ):
                    if s_hash in environment.iter_all_states():
                        self.xy_s_hashD[s_hash] = (x,y) # index=s_hash: value=x,y location (with (0,0) at Lower Left)

                        self.s_xyD[(x,y)] = s_hash  # index=(x,y):  value=s_hash
    
        # self.rev_lookupD = {} # index=s_hash, value=(row, col) in s_hash_rowL
        for irow, rowL in enumerate(self.s_hash_rowL):
            for jcol, s_hash in enumerate(rowL):
                self.rev_lookupD[s_hash] = (irow, jcol)
    
    def get_row_col(self, s_hash):
        """return the (row, col) location of s_hash within s_hash_rowL"""
        return self.rev_lookupD.get( s_hash, (None, None) )
    
    def get_s_hash_xy(self, s_hash):
        return self.xy_s_hashD.get(s_hash, None) # returns None if no (x,y) for s_hash

    def s_hash_print(self, none_str='*'):
        
        rows_outL = []
        for row in self.s_hash_rowL:
            outL = []
            for s_hash in row:
                if not self.environment.is_legal_state( s_hash ):
                    if is_literal_str( s_hash ):
                        outL.append( s_hash[1:-1] )
                    else:
                        outL.append( none_str )
                else:
                    if s_hash in self.named_s_hashD:
                        outL.append( self.named_s_hashD[s_hash]  )
                    else:
                        outL.append( str(s_hash)  )
            rows_outL.append( outL )
        
        if rows_outL:
            lmaxL = print_string_rows( rows_outL, 
                                       row_tickL=self.row_tickL, const_col_w=True,
                                       col_tickL=self.col_tickL,
                                       header=self.environment.name, 
                                       y_axis_label=self.y_axis_label,
                                       x_axis_label='State-Hash')
            return lmaxL # return the number of characters in each row.
        else:
            return []

    def param_print(self, paramD, 
                    row_tickL=None, const_col_w=True,
                    col_tickL=None, 
                    header='', 
                    x_axis_label='', y_axis_label='',
                    none_str='*'):
        """
        parameter values are in dictionary paramD
        paramD index=s_hash, value=string
        """
        rows_outL = []
        for row in self.s_hash_rowL:
            outL = []
            for s_hash in row:
                if (s_hash in paramD) and self.environment.is_legal_state( s_hash ):
                    outL.append( str( paramD[s_hash] )  )
                else:
                    if is_literal_str( s_hash ):
                        outL.append( s_hash[1:-1] )
                    elif s_hash in self.named_s_hashD:
                        outL.append( self.named_s_hashD[s_hash] )
                    else:
                        outL.append( none_str )
                    
            rows_outL.append( outL )
        
        if row_tickL is None:
            row_tickL = self.row_tickL
        
        if col_tickL is None:
            col_tickL = self.col_tickL
            
        if not x_axis_label:
            x_axis_label = self.x_axis_label
            
        if not y_axis_label:
            y_axis_label = self.y_axis_label
        
        if rows_outL:
            lmaxL = print_string_rows( rows_outL, row_tickL=row_tickL, 
                                       const_col_w=const_col_w,
                                       col_tickL=col_tickL,
                                       header=header, 
                                       x_axis_label=x_axis_label,
                                       y_axis_label=y_axis_label)
            return lmaxL # return the number of characters in each row.
        else:
            return []

    def s_hash_diagram(self, save_name='', basic_color='skyblue', do_show=False,
                       none_str='*', inp_colorD=None, pad=0.05, scale=1.0, h_over_w=1.0 ):
        """
        Create a PNG file of layout
        Use matplotlib to create a color-coded diagram.
        
        if inp_colorD is provided, it has, index=action, value=color string.
        pad determines the amount of white space between state rectangles.
        """
        
        if not got_matplotlob:
            print('WARNING... Need matplotlib to create a diagram... it failed to import.')
            return
        
        local_colorD = {}
        if self.colorD is not None:
            local_colorD.update( self.colorD )
        if inp_colorD is not None:
            local_colorD.update( inp_colorD )

        if self.basic_color:
            basic_color = self.basic_color
        
        #colorL = ['r','g','b','m','c','y',
        #          'darkcyan','deepskyblue','darkorange','brown','deeppink',
        #          'maroon','crimson','seagreen','fuchsia','darkviolet' ]
        
        Ncols = len( self.s_hash_rowL[0] )
        Nrows = len( self.s_hash_rowL )

        #fig, axs = plt.subplots()
        #fig.set_size_inches(Ncols+1, Nrows+1)
        
        w_lr = 1.0
        h_tb = 1.0
        fig = plt.figure( figsize=( scale*(Ncols+w_lr), h_over_w*scale*(Nrows+h_tb)) )
        
        axs = fig.add_axes()
        
        plt.axes()
        
        alignment = {'horizontalalignment': 'center', 'verticalalignment': 'center'}
        font = FontProperties()
        font.set_size('large')
        font.set_family('fantasy')
        font.set_style('normal')
        
        d = 1.0 - pad
        d2 = d / 2.0
        
        for irow,row in enumerate(self.s_hash_rowL):
            outL = []
            x = Nrows - irow - 1
            for jcol,s_hash in enumerate(row):
                
                if s_hash in local_colorD:
                    c = local_colorD[s_hash]
                else:
                    c = basic_color
                
                if self.environment.is_legal_state( s_hash ):
                    if s_hash in self.named_s_hashD:
                        s = self.named_s_hashD[s_hash]
                    else:
                        s = str( s_hash )
                    t = plt.text(jcol+d2, x+d2, s, fontproperties=font,**alignment)
                        
                    #      Rectangle(  (x,y),    width,   height)
                    rect = Rectangle((jcol,x), d,   d, fc=c, alpha=0.6, edgecolor=c)
                    plt.gca().add_patch( rect )
                        
                else:
                    if is_literal_str( s_hash ):
                        s = s_hash[1:-1]
                        t = plt.text(jcol+d2, x+d2, s, fontproperties=font,**alignment)
                    else:
                        rect = Rectangle((jcol,x), d,   d, fc='lemonchiffon', alpha=0.5, edgecolor='gray')
                        plt.gca().add_patch( rect )
                    
                             
    
        plt.xlim(0, Ncols)
        plt.ylim(0, Nrows)
        #plt.axis('off')
        
        if self.col_tickL is None:
            plt.xticks([])
        else:
            plt.xticks( [i+0.5 for i in range(len(self.col_tickL))], 
                        [str(ct) for ct in self.col_tickL] )

        if self.row_tickL is None:
            plt.yticks([])
        else:
            plt.yticks( [i+0.5 for i in range(len(self.row_tickL))], 
                        reversed([str(rt) for rt in self.row_tickL]) )
        
        
        plt.box(False)
        
        plt.title( self.environment.name )

        if self.x_axis_label:
            plt.xlabel( self.x_axis_label )
            
        if self.y_axis_label:
            plt.ylabel( self.y_axis_label )
            
        plt.tight_layout()


        if save_name:
            if save_name.lower().endswith('.png'):
                fig.savefig( save_name )
            else:
                fig.savefig( save_name + '.png' )
                
        if do_show:
            plt.show()
        

if __name__ == "__main__": # pragma: no cover
    
    from introrl.mdp_data.sample_gridworld import get_gridworld
    gridworld = get_gridworld()
    
    gridworld.layout.s_hash_diagram( save_name='sample_gridworld', none_str='*', do_show=True,
                                       inp_colorD={'Goal':'g', 'Pit':'r', 'Start':'b'}, 
                                       pad=0.05, scale=0.75)

