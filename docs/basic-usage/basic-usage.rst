ATTgt
-----

Basic Example
*************

.. ipython:: python

    from differences import load_data, ATTgt

    dataset = load_data.mpdta()

    # compute group-time ATT
    att_gt = ATTgt(
        data=dataset["data"],
        cohort_name=dataset["cohort_name"],
    )

.. ipython-with-reprs:: reg,dr,std_ipw-mle

    att_gt.fit("lemp ~ lpop")

    att_gt.aggregate("event")
    att_gt.aggregate("time")
    att_gt.aggregate("cohort")
    att_gt.aggregate("simple")


Example with simulated data
***************************

.. ipython:: python
    :okwarning:

    from differences import simulate_data, ATTgt

    panel_data = simulate_data()  # generate data

    att_gt = ATTgt(data=panel_data, cohort_name='cohort')


.. ipython:: python
    :okwarning:

    att_gt.fit(formula='y ~ x0')

    att_gt.aggregate('time')

    att_gt.aggregate('event')

    att_gt.aggregate('cohort')

    att_gt.aggregate('simple')

    att_gt.aggregate('event', overall=True)

Heterogeneity and triple difference
===================================

.. ipython:: python
    :okwarning:

    # heterogeneity

    panel_data = simulate_data(samples=3)

    att_gt = ATTgt(data=panel_data, cohort_name='cohort')

    att_gt.fit(formula='y', split_sample_by='samples')

    att_gt.aggregate('event')

    att_gt.aggregate('simple')

    # triple difference
    att_gt.aggregate('time', difference=['samples = 1', 'samples = 2'])

Multi-valued Treatment
======================


.. ipython:: python
    :okwarning:

    # multi-valued treatment
    panel_data = simulate_data(intensity_by=2)  # generate data

    att_gt = ATTgt(data=panel_data, cohort_name='cohort', strata_name='strata')

    att_gt.fit(formula='y', n_jobs=1)

    att_gt.aggregate('event')

    att_gt.aggregate('simple')

    att_gt.aggregate('event', difference=[0, 1], boot_iterations=5000)

