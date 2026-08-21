# CHANGELOG

<!-- version list -->

## v2.0.0 (2026-08-21)

### Bug Fixes

- Broken tests using old sampler API surface
  ([`2ffd625`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/2ffd6253b374aed20486dff07e2eaaab7b5374ba))

- Dtype and device parity for mMMD loss
  ([`d6b5f84`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/d6b5f848d1b5d90a2c3305c4602f9559f861bace))

- Remove redundant/broken wrappers to HFHub export methods
  ([`31e7f01`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/31e7f01d9f8756dc5c9129d63d70027cf9dee0ab))

- Respect new shape behaviour of single-float tensors
  ([`e5cdc53`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/e5cdc53dcdbb71bac87c1017ebfc0986f595d570))

### Build System

- Bump deps
  ([`05403d3`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/05403d3ea272190e748392b7d65d7caf1c75a628))

- Configure ruff to ignore B008
  ([`b0954b7`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/b0954b78ed3e096f86529b740311a68ac610fb81))

- Remove ty dev dependency for parity with CI
  ([`a708df8`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/a708df8f94b726620bdca4bc644eff07d977179b))

### Chores

- Import linting
  ([`39ec59e`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/39ec59eefbd45d65956e056fb663cc1709d3a904))

### Documentation

- Fix docs deployment action
  ([`a773465`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/a773465a663cbd4bbd95c5b86a13368a5806b305))

- Fix docstring dimension error in Sampler
  ([`3c98771`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/3c987719537db4f2b40db6c3eaa675c80a2792b1))

### Features

- Adjust sampler API to avoid bare tokenizer calls in pipeline
  ([`b4a4c15`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/b4a4c15bfc42837742ff875f3040cabc756c2825))


## v1.0.0 (2026-07-27)

- Initial Release
