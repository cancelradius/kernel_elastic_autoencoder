import pytest
import torch.nn.functional as F


class TestTop1Sampler:
    @pytest.mark.parametrize("batch_size,seq_length", [(2, 5), (4, 10)])
    def test_reconstruct(self, batch_size, seq_length, top_1_sampler, randlong):
        sampler = top_1_sampler()
        ground_truth = randlong(
            0, sampler.tokenizer.vocab_size, (batch_size, seq_length)
        )
        one_hot = F.one_hot(ground_truth, sampler.tokenizer.vocab_size)
        assert sampler(one_hot, skip_special_tokens=False) == sampler.tokenizer.decode(
            ground_truth, skip_special_tokens=False
        )


class TestKAEPipeline:
    def test_completion(self, kae_pipeline, randn, sample_dataframe):
        pipe = kae_pipeline()
        ds = sample_dataframe()
        ds["seq"] = ""
        pipe.completion(
            latents=randn(
                (
                    5,
                    pipe.model.config_typed.common.embedding_dim
                    * pipe.model.config_typed.common.pooling_dim,
                )
            ),
            dataset=ds,
            seq_feature="seq",
            cond_features=[
                "cond1",
                "cond2",
            ],
            device=None,
        )
