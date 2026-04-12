<img src="./docs/imgs/logo/bw/logo_name_bw.png" alt="drawing" width="200"
style="display: block;margin-left: auto;margin-right: auto;width: 80%;"/>

difference-in-differences estimation and inference for Python

**For the following use cases**

- Balanced panels, unbalanced panels & repeated cross-section
- Two + Multiple time periods
- Fixed + Staggered treatment timing
- Binary + Multi-Valued treatment
- Heterogeneous treatment effects & triple difference
- Parallelized estimation & multiplier bootstrap (joblib, via `n_jobs`)

see the [Documentation](https://bernardodionisi.github.io/differences/latest/) for more details.

## Installing

The latest release can be installed using pip

```bash
pip install differences
```

requires Python >= 3.10

## Quick Start

### ATTgt

the ATTgt class implements the estimation procedures suggested by [Callaway and Sant'Anna (2021)
](https://www.sciencedirect.com/science/article/abs/pii/S0304407620303948), [Sant'Anna and Zhao
(2020)](https://www.sciencedirect.com/science/article/abs/pii/S0304407620301901) and the
multi-valued treatment case discussed
in [Callaway, Goodman-Bacon & Sant'Anna (2021)](https://arxiv.org/abs/2107.02637)

```python
import differences

df = differences.generate_data()  # requires the [all] extra: pip install differences[all]

att_gt = differences.ATTgt(data=df, cohort_column='cohort')

att_gt.fit(formula='y', n_jobs=-1)

att_gt.aggregate('event')
```

*differences* ATTgt was inspired by
- Callaway & Sant'Anna's [did](https://github.com/bcallaway11/did)
- Sant'Anna and Zhao's [DRDID](https://github.com/pedrohcgs/DRDID)
