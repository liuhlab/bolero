import torch
import pytest

from bolero.tl.model.flow._velocity_field import ConditionalVelocityField

x_test = torch.ones((10, 5)) * 10
t_test = torch.ones((10, 1))
x_0_test = x_test + 2.0
cond = {"pert1": torch.ones((1, 2, 3))}


class TestVelocityField:
    @pytest.mark.parametrize("decoder_dims", [(32, 32), (2, 2)]) # , (2, 2)
    @pytest.mark.parametrize("hidden_dims", [(32, 32), (2, 2)]) # , (2, 2)
    @pytest.mark.parametrize("layer_norm_before_concatenation", [False, True]) # True,
    @pytest.mark.parametrize("linear_projection_before_concatenation", [False, True]) # True,
    @pytest.mark.parametrize("condition_mode", ["deterministic", "stochastic"]) #
    @pytest.mark.parametrize("conditioning", ["concatenation"])
    def test_velocity_field_init(
        self,
        hidden_dims,
        decoder_dims,
        layer_norm_before_concatenation,
        linear_projection_before_concatenation,
        condition_mode,
        conditioning,
    ):
        layers_before_pool = {"pert1": ({"input_dim": 3},)}
        layers_after_pool = ({"layer_type": "mlp", "dims": (32, 32, 12), "input_dim": 3},)
        vf = ConditionalVelocityField(
            output_dim=5,
            emb_input_dim=5,
            max_combination_length=2,
            condition_mode=condition_mode,
            condition_embedding_dim=12,
            hidden_dims=hidden_dims,
            decoder_dims=decoder_dims,
            layer_norm_before_concatenation=layer_norm_before_concatenation,
            linear_projection_before_concatenation=linear_projection_before_concatenation,
            conditioning=conditioning,
            layers_before_pool=layers_before_pool,
            layers_after_pool=layers_after_pool
        )
        assert vf.output_dims == decoder_dims + (5,)
        assert vf.conditioning == conditioning

        vf.train()
        encoder_noise = torch.randn((x_test.shape[0], vf.condition_embedding_dim))
        out, out_mean, out_logvar = vf(t_test, x_test, cond, encoder_noise)

        assert out.shape == (10, 5)
        assert out_mean.shape == (1, 12)
        assert out_logvar.shape == (1, 12)

        out_mean, out_logvar = vf.get_condition_embedding(cond)
        assert out_mean.shape == (1, 12)
        assert out_logvar.shape == (1, 12)

    # @pytest.mark.parametrize("condition_mode", ["deterministic", "stochastic"])
    # @pytest.mark.parametrize("conditioning", ["concatenation"])
    # def test_velocityfield_conditioning_kwargs(self, condition_mode, conditioning):
    #     kwargs = {}
    #     vf = ConditionalVelocityField(
    #         output_dim=5,
    #         max_combination_length=2,
    #         condition_mode=condition_mode,
    #         condition_embedding_dim=12,
    #         hidden_dims=[2, 2],
    #         decoder_dims=[2, 2],
    #         conditioning=conditioning,
    #         **kwargs,
    #     )
    #     vf.train()
    #     encoder_noise = torch.randn((x_test.shape[0], vf.condition_embedding_dim))
    #     out, out_mean, out_logvar = vf(
    #         t_test,
    #         x_test,
    #         cond,
    #         encoder_noise
    #     )

    #     assert out.shape == (10, 5)
    #     assert out_mean.shape == (1, 12)
    #     assert out_logvar.shape == (1, 12)
