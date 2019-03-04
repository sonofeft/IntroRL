
.. chapter_9

Chapter 9 On-Policy Prediction with Approximation
=================================================

This chapter begins function approximation by looking at on-policy prediction.

Given a policy, use function approximation to estimate the state-value function.

Gradient MC Prediction
----------------------

The pseudo code for Monte Carlo prediction is taken from page 201 of
`Sutton & Barto <http://incompleteideas.net/book/the-book-2nd.html>`_ and is shown below.

.. image:: _static/mc_gradient_pseudocode.jpg

Figure 9.1
----------

Figure 9.1 on page 204 of `Sutton & Barto <http://incompleteideas.net/book/the-book-2nd.html>`_
uses the above Monte Carlo Prediction on the 1000-state random walk to aggregate the 1000 states
into 10 approximate states.

The left image is taken from `Sutton & Barto <http://incompleteideas.net/book/the-book-2nd.html>`_
and the right image is the result of the **IntroRL** script
`Figure 9.1 Code <./_static/colorized_scripts/examples/chapter_9/plot_randwalk1000.html>`_

.. image:: _static/fig_9_1_sutton.jpg
    :width: 55%

.. image:: _static/figure_9_1.png
    :width: 44%

Each of the three curves displayed on the **IntroRL** chart are created by a different support script.

`Figure 9.1 True Value <./_static/colorized_scripts/examples/chapter_9/calc_rw1000_trueval.html>`_

`Figure 9.1 MC Approximation Value <./_static/colorized_scripts/examples/chapter_9/mc_rw1000_eval.html>`_

`Figure 9.1 Distribution Scale <./_static/colorized_scripts/examples/chapter_9/calc_mu_rw1000.html>`_
