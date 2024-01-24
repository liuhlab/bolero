# bolero

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/lhqing/bolero/test.yaml?branch=main
[link-tests]: https://github.com/lhqing/bolero/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/bolero

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].

## Installation

You need to have Python 3.10 or newer installed on your system. If you don't have
Python installed, we recommend installing [Miniforge](https://github.com/conda-forge/miniforge).

```bash
# 1. Download the environment YAML file
wget https://raw.githubusercontent.com/lhqing/commons/main/envs/bolero_env.yaml

# 2. Create a environment named bolero
mamba env create -f bolero_env.yaml
# OR if you use conda
# conda env create -f bolero_env.yaml
# Note that conda can be very slow in solving complex dependencies

# 3. Install this package
pip install bolero

# or install the package with dev mode
git clone https://github.com/lhqing/bolero.git
cd bolero
pip install -e ".[dev,test]"
```

## Release notes

See the [changelog][changelog].

## Contact

If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/lhqing/bolero/issues
[changelog]: https://bolero.readthedocs.io/latest/changelog.html
[link-docs]: https://bolero.readthedocs.io
[link-api]: https://bolero.readthedocs.io/latest/api.html
