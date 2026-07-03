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

`bolero` uses [pixi](https://pixi.sh) to manage its environment. Install pixi, then:

```bash
git clone https://github.com/liuhlab/bolero.git
cd bolero
pixi install            # runtime env; or `pixi install -e dev` for tests/docs/lint/notebooks
```

See [Installation](docs/installation.md) for GPU/CUDA and optional-dependency (`ray`,
`flash-attn`) notes.

## Release notes

See the [changelog][changelog].

## Contact

If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/lhqing/bolero/issues
[changelog]: https://bolero.readthedocs.io/latest/changelog.html
[link-docs]: https://liuhlab.github.io/bolero/
[link-api]: https://liuhlab.github.io/bolero/
