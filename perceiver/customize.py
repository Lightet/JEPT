from typing import Tuple, Optional

import torch
from einops import repeat

from perceiver.adapter import InputAdapter, OutputAdapter, TrainableQueryProvider
from perceiver.position import FourierPositionEncoding
from perceiver.modules import PerceiverEncoder, PerceiverDecoder, PerceiverIO


class Seq2DInputAdapter(InputAdapter):
    def __init__(self,
                 sequence_shape: Tuple[int, int],
                 num_pos_enc_channels: int,
                 num_input_channels: int,
                 use_cls_token: bool = True,
                 pos_enc_type: Optional[str] = "cat"):
        """
        Adapter for 2d sequences input_sequence
        :param sequence_shape: Shape of the input_sequence tensor.
        :param num_pos_enc_channels: Number of position encoding channels.
        :param num_input_channels: Number of input_sequence channels.
        :param use_cls_token: Whether to use a [cls] token in the input_sequence tensor from VIT model.
        :param pos_enc_type: Type of position encoding to use. Options: "cat", "add".
        """
        assert num_pos_enc_channels % 4 == 2, "Number of position encoding channels must be even and divisible by 4."
        num_frequency_bands = num_pos_enc_channels // 4
        position_encoding = FourierPositionEncoding(input_shape=sequence_shape, num_frequency_bands=num_frequency_bands)
        super().__init__(num_input_channels=num_input_channels + num_pos_enc_channels)
        self.position_encoding = position_encoding
        self.pos_enc_type = pos_enc_type
        if use_cls_token:
            self.cls_pos_enc = torch.nn.Parameter(torch.randn(num_pos_enc_channels), requires_grad=True)

    def forward(self, input_sequence, cls_token=True):
        """
        Forward pass through the frame hidden adapter.
        :param input_sequence: Hidden states from a Vision Transformer model.
        :param cls_token: Whether a [cls] token is present in the frame hidden states.
        :return:
        """
        b, *_ = input_sequence.shape
        pos_enc = self.position_encoding(b)  # Position encoding (b, seq_len, num_pos_enc_channels)
        print(pos_enc.shape)
        if cls_token:
            cls_pos_enc = repeat(self.cls_pos_enc, 'c -> b 1 c', b=b)
            pos_enc = torch.cat((cls_pos_enc, pos_enc), dim=1)
        if self.pos_enc_type == "cat":
            return torch.cat((input_sequence, pos_enc), dim=-1)
        elif self.pos_enc_type == "add":
            return input_sequence + pos_enc


class Seq2DOutputAdapter(OutputAdapter):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Seq2DPerceiver(PerceiverIO):
    def __init__(self, _config):
        input_adapter = Seq2DInputAdapter(**_config['input_adapter'])
        encoder = PerceiverEncoder(input_adapter=input_adapter, **_config['encoder'])

        output_adapter = Seq2DOutputAdapter()
        output_query_provider = TrainableQueryProvider(**_config['output_query_provider'])
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            output_query_provider=output_query_provider,
            **_config['decoder']
        )
        super().__init__(encoder=encoder, decoder=decoder)

    def forward(self, x, pad_mask=None):
        x_latent = self.encoder(x, pad_mask)
        return self.decoder(x_latent)


if __name__ == "__main__":
    from torchinfo import summary

    import yaml

    config = yaml.safe_load(open("../configs/metaworld/frame_adapter.yaml"))
    model = Seq2DPerceiver(config)
    print(model)
    info = summary(model, (100, 197, 786))
    print(info)