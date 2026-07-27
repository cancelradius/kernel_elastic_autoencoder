## `kernel_elastic_autoencoder`

`kernel_elastic_autoencoder` is a library implementing the architecture and techniques described
in [this publication](https://doi.org/10.1093/pnasnexus/pgae168) from Li et al. I am not affiliated with the authors of
the original paper, and this implementation is provided as-is, with no guarantee of completeness.

### Installation

`kernel_elastic_autoencoder` can be installed with pip, and regular builds are provided on PyPI:

    pip install kernel_elastic_autoencoder

> Please note that `torch` is not included as a dependency due to its many hardware-accelerator-dependent versions, so
> take care to install the appropriate version manually.

Distribution builds are also provided here on GitHub Releases. New builds are triggered by the CD Action, so they will 
be made available as soon as a new PR is merged to `main`.

Alternatively, for development purposes, `kernel_elastic_autoencoder` may be installed from source provided here. Builds
and deps are managed with `poetry`.

### Documentation

API documentation is generated with `pdoc` and covers the `__all__`-exported interfaces. It is available on 
[GitHub Pages](https://cancelradius.github.io/kernel_elastic_autoencoder).