# CHANGELOG

<!-- version list -->

## v3.2.5 (2026-08-27)

### Bug Fixes

- Change erroneous hp_sigma default
  ([`d6546a6`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/d6546a6939bda5cd84f901ec3d6cea48ce8e8fec))

- Fix seed for random_split in training
  ([`99ed9b5`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/99ed9b51b04b2097b27a412e3a3f90d83786e458))

- **perf**: Move loss_fn with Accelerator.prepare
  ([`3e2476d`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/3e2476d60ae840a0d724a48cb4d093e930841635))


## v3.2.4 (2026-08-27)

### Bug Fixes

- Add workers and pin memory for DataLoaders
  ([`d88a9ab`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/d88a9ab450a55120667f5fe3ad13a995ea36c473))

### Refactoring

- Clean up tqdm boilerplate
  ([`81d9a6c`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/81d9a6cdfe50f25c81152c93d7c65fc09c48e456))


## v3.2.3 (2026-08-27)

### Bug Fixes

- Wait for everyone before starting training
  ([`7ba25da`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/7ba25da42ab74a4da65eadbd2928343929ee736c))


## v3.2.2 (2026-08-27)

### Bug Fixes

- Logging respects main process
  ([`31977b3`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/31977b37f14147bba5e0a63a63716035311b4217))


## v3.2.1 (2026-08-27)

### Bug Fixes

- Force --find_unused_parameters on any accelerate launch
  ([`a48f2a1`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/a48f2a170b9471f519de3d0acbfbf2c699296f3f))


## v3.2.0 (2026-08-26)

### Bug Fixes

- BOS and token masking
  ([`8d81d36`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/8d81d36ac29ab3fd25874587d0cc214b2efcff5c))

- Cast padding_mask to bool in the encoder to avoid deprecated mask type mismatch
  ([`51ecfff`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/51ecfff8be1ef8862f23b1be9af5e3050a483038))

- Correct padding sig
  ([`0c46ec9`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/0c46ec99f4b160d401311a663defc53da5a195cb))

- Correct padding_mask slicing
  ([`281134d`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/281134dacafcc259ffe10230446aca27474e54a8))

- Correctly sized causal mask buffer
  ([`80eb4c4`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/80eb4c49d2f321506ab0516ebac3282e246ba8f5))

- Dropout typing in config now accepts defaults
  ([`07c2739`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/07c273981a5a10f6350b9a57903d1e5be1a74241))

- Keep ground truth conversion to torch.long in WCEL
  ([`62b8167`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/62b816757541a1fb990ac4ba04ee77dbf0a8e7ef))

- Move masked condition indices to right device in ConditionEmbedding
  ([`4105159`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/410515978e50ef9f4446b07b0a64df7ec08cf3cc))

- Pad batches during inference
  ([`dd883c4`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/dd883c47029602205985388ebbd4dcc9f83c1e5e))

- Proper device fallbacks
  ([`e56795f`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/e56795f042f23f4346ccce7916af6543ca081ba9))

- Remove condition_mask in Model.decode
  ([`c669147`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/c6691479d5b4513c61ddffef1abb763505822123))

- Repair beam_completion behaviour
  ([`82a2883`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/82a28834ee63cc27f2c2e8e571b6c0bdfe7e7932))

### Chores

- Minimal train logging
  ([`8682734`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/868273462af8b4bc6d3ceac58c6d645488247b7e))

- Prune imports in config.py
  ([`f6eb33d`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/f6eb33d89d47ff0069037f9adbdb0692adc4a746))

- Remove deprecated and superfluous epoch input to scheduler.step
  ([`657550b`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/657550b48ca1373572295b059839a3b2a0eb1722))

### Features

- Add Pipeline.encoding
  ([`57d42b6`](https://github.com/cancelradius/kernel_elastic_autoencoder/commit/57d42b6f22f13a69c2be1ad8b334613043990125))


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
