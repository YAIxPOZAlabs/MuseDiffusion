from transformers import FNetConfig, BertConfig
from transformers.models.fnet.modeling_fnet import FNetEncoder
# from transformers import BertEncoder
# from transformers.models.bert.modeling_bert import FNetModel
import torch

import numpy as np
import torch as th
import torch.nn as nn

from ..model import FNetHybrid
from .utils.nn import (
    SiLU,
    linear,
    timestep_embedding,
)


class TransformerNetModel(nn.Module):
    """
    The full Transformer model with attention and timestep embedding.

    :param input_dims: dims of the input Tensor.
    :param output_dims: dims of the output Tensor.
    :param hidden_t_dim: dims of time embedding.
    :param dropout: the dropout probability.
    :param vocab_size: the size of vocabulary
    """

    def __init__(
            self,
            input_dims,
            output_dims,
            hidden_t_dim,
            vocab_size,
            fnet_hidden_dim,  # for FNet
            fnet_intermediate_dim,  # for FNet
            seq_len,  # for FNet
            num_fnet_layers,  # for FNet
            dropout=0,
            logits_mode=1,
            config=None,  # TODO
            use_attention=False,
    ):
        super().__init__()

        if config is None:
            ## FNet Config
            config = FNetConfig()  # AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout
            config.hidden_size = fnet_hidden_dim
            config.intermediate_size = fnet_intermediate_dim
            config.max_position_embeddings = seq_len
            config.vocab_size = vocab_size
            config.num_hidden_layers = num_fnet_layers
            config.pad_token_id = 0
            config.eos_token_id = 1
            config.type_vocab_size = 2

            attention_config = BertConfig()
            attention_config.num_attention_heads = 8
            attention_config.hidden_size = fnet_hidden_dim
            attention_config.intermediate_size = fnet_intermediate_dim

        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.dropout = dropout
        self.logits_mode = logits_mode
        self.hidden_size = config.hidden_size

        self.word_embedding = nn.Embedding(vocab_size, self.input_dims, padding_idx=0)
        self.type_id_embedding = nn.Embedding(2, self.input_dims)

        self.lm_head = nn.Linear(self.input_dims, vocab_size)
        with th.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        if self.input_dims != config.hidden_size:
            self.input_up_proj = nn.Sequential(nn.Linear(input_dims, config.hidden_size),
                                               nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))

        self.input_transformers = FNetHybrid(config,attention_config,use_attention)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if self.output_dims != config.hidden_size:
            self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                                  nn.Tanh(), nn.Linear(config.hidden_size, self.output_dims))

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2:  # standard cosine similarity
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1))  # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError

    def forward(self, x, timesteps,model_kwargs=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))
        input_ids_mask = model_kwargs['input_mask'].to(x.device)

        if self.input_dims != self.hidden_size:
            emb_x = self.input_up_proj(x)
        else:
            emb_x = x

        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length]
        # print(emb_x.shape, emb_t.shape, self.position_embeddings)
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1) + self.type_id_embedding(input_ids_mask)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        input_trans_hidden_states = self.input_transformers(emb_inputs,model_kwargs)

        if self.output_dims != self.hidden_size:
            h = self.output_down_proj(input_trans_hidden_states)
        else:
            h = input_trans_hidden_states
        h = h.type(x.dtype)
        return h
