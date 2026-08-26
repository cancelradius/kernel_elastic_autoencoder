# CHANGELOG

<!-- version list -->

## v3.1.2 (2026-08-25)

### Bug Fixes

- Always use padding_idx 0 for condition masks
  ([`cdc4bec`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/cdc4bec650445af0626856a67518b9e1c13a0437))

- Force Tokenizer protocol compatibility with transformers.PreTrainedTokenizerBase
  ([`ae90e20`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/ae90e2074194e1eec258b84a2933abc591c99ddd))

- Move mMMD samples to correct device
  ([`a48d8b0`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/a48d8b090f21256fb539602079eb4e5857af7154))

- Track last_epoch via scheduler
  ([`7c0e57b`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/7c0e57b695089c87200f5ccfd667ad5e901afbbf))

### Chores

- Update tokenizer.encode sig to reflect Protocol changes
  ([`c3a9800`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/c3a9800384cf9a5e66c2a953af1721289b771470))


## v3.1.1 (2026-08-25)

### Bug Fixes

- Validate defaults for optimizer and scheduler config functions
  ([`443b275`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/443b27516aec14d397169fa3c85a157f395166ef))


## v3.1.0 (2026-08-24)

### Documentation

- Add beam_completion docstring
  ([`4af44f9`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/4af44f9ed16ac14e90305f53a7f8fdd951cf38ae))

### Features

- Add beam search algorithm to Pipeline API
  ([`c09abb4`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/c09abb4970cd68014e2765ab3e9a54fa327eabce))

### Refactoring

- Flatten Pipeline.completion and refactor away private methods
  ([`a5da582`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/a5da58284830d6a5fb331e52b3a4b3b4f1417774))


## v3.0.1 (2026-08-23)

### Bug Fixes

- Move input_ids to correct device before Pipeline inference
  ([`3c88c13`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/3c88c132559d4b273a63c069a11671d45ae1f6d2))

### Build System

- Temp lower cov-fail-under to allow CI to proceed
  ([`c639e19`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/c639e192c3eb4ba185fedde5ccb9fa26d7a34c79))

### Documentation

- Remove erroneous reference to Pipeline.completion.outputs
  ([`f177a1e`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/f177a1e63cd5597e1e0696c50d74623f71f5a408))


## v3.0.0 (2026-08-23)

### Bug Fixes

- Training checkpoint epoch tracking and correct load
  ([`bb75df2`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/bb75df248e606c00b53dc3fd64d05c31d0faaca7))

### Chores

- Clean imports from previous breaking API change
  ([`29d64cb`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/29d64cbeec2cb0ec94b8cb119a829fa719b880bc))

- Remove orphaned __init__ imports from Trainer refactor
  ([`072ad13`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/072ad1398578461debb90bcdfb360f52f7f5611a))

- Remove orphaned SimpleNamespace import in pipeline.py
  ([`3df9cd2`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/3df9cd27adf005ccde2378ec570500c4280a182c))

### Documentation

- Add missing docstring to embed_conditions
  ([`35e44f3`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/35e44f338b22cfce4f37bc73ab348e32f7718dca))

### Features

- **api**: Remove Collated schema and streamline training input
  ([`fb7f436`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/fb7f43605f3cd7e1357b96770a6ade48543b1d1a))

- **api**: Streamline API by changing sampler and collator interfaces
  ([`6e190ea`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/6e190ea3342ca64f350bbc91d2f4889619e25b77))

### Refactoring

- Refactor of Trainer and integration of HF Accelerate
  ([`d9e7ddf`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/d9e7ddffda0922e90603a86ff20880f20728e221))

### Testing

- Remove dummy tokenizer and related tests
  ([`5631f75`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/5631f757d9187c855131ff4bfe226892d3d6912e))


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
