from differences import simulate_data, ATTgt

panel_data = simulate_data()  # generate data

att_gt = ATTgt(data=panel_data, cohort_name='cohort')

att_gt.fit(formula='y ~ x0')

att_gt.aggregate('time')

att_gt.aggregate('event')

att_gt.aggregate('cohort')

att_gt.aggregate('simple')

att_gt.aggregate('event', overall=True)

# heterogeneity

panel_data = simulate_data(samples=3)

att_gt = ATTgt(data=panel_data, cohort_name='cohort')

att_gt.fit(formula='y', split_sample_by='samples')

att_gt.aggregate('event')

att_gt.aggregate('simple')

# triple difference
att_gt.aggregate('time', difference=['samples = 1', 'samples = 2'])

# multi-valued treatment
panel_data = simulate_data(intensity_by=2)  # generate data

att_gt = ATTgt(data=panel_data, cohort_name='cohort', strata_name='strata')

att_gt.fit(formula='y', n_jobs=1)

att_gt.aggregate('event')

att_gt.aggregate('simple')

att_gt.aggregate('event', difference=[0, 1], boot_iterations=5000)
