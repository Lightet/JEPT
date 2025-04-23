import yaml
from torch import nn
from transformers import AutoModel, AutoImageProcessor
from x_transformers import ContinuousTransformerWrapper, Encoder

from perceiver import Seq2DPerceiver


class FrameEncoder(nn.Module):
    """
    Frame Encoder class to encode frames using Vision Transformer (ViT) model and an adapter.
    """

    def __init__(self, model_name='google/vit-base-patch16-224-in21k'):
        """
        :param model_name: model name of the Vision Transformer (ViT) model
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)

    def encode(self, frames):
        """
        Encode frames using Vision Transformer (ViT) model and an adapter.
        :param frames: list of frames to encode
        :return: encoded frames
        """
        # Extract pixel values and prepare input_sequence tensor
        inputs = self.processor(images=frames, return_tensors='pt')
        # TODO: Check if this is correct

        # Forward pass through the ViT model
        outputs = self.model(**inputs)  # (batch_size, seq_len, hidden_size)
        return outputs.last_hidden_state

    def forward(self, frames):
        return self.encode(frames)


class Resampler(nn.Module):
    """
    Resampler class to resample vectors using a perceiver.
    """

    def __init__(self, config_path='./configs/metaworld/prompt_resampler.yaml'):
        super().__init__()
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.resampler = Seq2DPerceiver(config)

    def forward(self, vectors, pad_mask=None):
        """
        :param vectors: list of vectors to resample
        :param pad_mask: padding mask tensor
        """
        return self.resampler(vectors, pad_mask)


class ActionPerceiver(nn.Module):
    """
    Action Head class to predict actions using a perceiver.
    """

    def __init__(self, config_path='./configs/metaworld/action_head.yaml'):
        super().__init__()
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.action_perceiver = Seq2DPerceiver(config)
        self.action_head = nn.Linear(config['hidden_size'], config['action_bins'])

    def forward(self, hidden_states):
        """
        :param hidden_states: list of hidden states to predict actions
        """
        action_embeddings = self.action_perceiver(hidden_states)
        action_logits = self.action_head(action_embeddings)
        return action_logits


class TransformerEncoder(nn.Module):
    """
    Encoder class to encode trajectories using a Transformer model.
    """

    def __init__(self, config_path='./configs/metaworld/trajectory_encoder.yaml'):
        super().__init__()
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.attn_layers = Encoder(**config['attn_layers'])
        self.encoder = ContinuousTransformerWrapper(attn_layers=self.attn_layers, **config['encoder'])

    def forward(self, inputs, padding_mask=None, attention_mask=None):
        """
        :param inputs: input_sequence tensor
        :param padding_mask: padding mask tensor
        :param attention_mask: attention mask tensor
        """
        return self.encoder(inputs, mask=padding_mask, attn_mask=attention_mask)
