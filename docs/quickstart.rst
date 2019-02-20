
.. quickstart

QuickStart
==========

Install IntroRL
---------------

The easiest way to install **IntroRL** is::

    pip install introrl
    
        OR on Linux
    sudo pip install introrl
        OR perhaps
    pip install --user introrl

In case of error, see :ref:`internal_pip_error`

.. _internal_source_install:

Installation From Source
------------------------

Much less common, but if installing from source, then
the best way to install introrl is still ``pip``.

After navigating to the directory holding **IntroRL** source code, do the following::

    cd full/path/to/introrl
    pip install -e .
    
        OR on Linux
    sudo pip install -e .
        OR perhaps
    pip install --user -e .
    
This will execute the local ``setup.py`` file and insure that the pip-specific commands in ``setup.py`` are run.

Running IntroRL
---------------

After installing, try the following simple code block to run a dynamic programming value iteration
on a simple gridworld.

.. code-block:: python
   
    from introrl.dp_funcs.dp_value_iter import dp_value_iteration
    from introrl.mdp_data.simple_grid_world import get_gridworld

    gridworld = get_gridworld()

    policy, state_value = dp_value_iteration( gridworld, do_summ_print=True,
                                              max_iter=1000, err_delta=0.001, 
                                              gamma=0.9)
   
The output should look similar to::

    Exited Value Iteration 
       iterations     = 4  (limit=1000)
       measured delta = 0.0
       gamma          = 0.9
       err_delta      = 0.001
       error limit    = 0.00011111111111111108
       STOP CRITERIA  = 0.11111111111111108

    ___ "Simple Grid World" State-Value Summary ___
     ==== Simple Grid World ====
    (0, 0) (0, 1) (0, 2) (0, 3)  ||  
    (1, 0)      * (1, 2) (1, 3)  ||  
    (2, 0) (2, 1) (2, 2) (2, 3)  ||  
     ======== State-Hash =======
         ___ Simple Grid World State-Value Summary, V(s) ___
                       0.81    0.9      1      0  ||  
                      0.729      *    0.9      0  ||  
                     0.6561  0.729   0.81  0.729  ||  
    ___ Policy Summary ___
        Nstate-actions=9
         ___ Simple Grid World Policy Summary ___
                       R   R   R   *  ||  
                       U   *   U   *  ||  
                       R   R   U   L  ||  
         _______________ Actions ________________
         ___ Simple Grid World Reward Summary ___
                        0  0  0  1  ||  
                        0  *  0 -1  ||  
                        0  0  0  0  ||  


.. _internal_pip_error:

pip Error Messages
------------------

If you get an error message that ``pip`` is not found, see `<https://pip.pypa.io/en/latest/installing.html>`_ for full description of ``pip`` installation.

There might be issues with ``pip`` failing on Linux with a message like::


    InsecurePlatformWarning
            or    
    Cannot fetch index base URL https://pypi.python.org/simple/

Certain Python platforms (specifically, versions of Python earlier than 2.7.9) have the InsecurePlatformWarning. If you encounter this warning, it is strongly recommended you upgrade to a newer Python version, or that you use pyOpenSSL.    

Also ``pip`` may be mis-configured and point to the wrong PyPI repository.
You need to fix this global problem with ``pip`` just to make python usable on your system.


If you give up on upgrading python or fixing ``pip``, 
you might also try downloading the introrl source package 
(and all dependency source packages)
from PyPI and installing from source as shown above at :ref:`internal_source_install`


