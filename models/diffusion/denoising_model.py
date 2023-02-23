"""backbone for denoising"""
from transformers.models.fnet.modeling_fnet import FNetEncoder
from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput
import torch.nn as nn

'''
Not used
'''


###########################################
## FNet Hybrid
###########################################
class FNetHybrid(nn.Module):
    def __init__(self,fnet_config,attention_config,use_attention=False) -> None:
        super().__init__()
        self.fnet = FNetEncoder(fnet_config)
        self.use_attention = use_attention
        if use_attention:
            self.attention_layer_1 = BertAttentionLayer(attention_config)
            self.attention_layer_2 = BertAttentionLayer(attention_config)

    def forward(self, x, model_kwargs):
        x = self.fnet(x).last_hidden_state
        if self.use_attention:
            x = self.attention_layer_1(x,attention_mask=model_kwargs['attention_mask'].to(x.device))
            x = self.attention_layer_2(x,attention_mask=model_kwargs['attention_mask'].to(x.device))
        return x


###########################################
## FNet
###########################################
class FNet(FNetEncoder):
    pass


###########################################
## Bert Attention Layer
###########################################
class BertAttentionLayer(nn.Module):
    def __init__(self, bert_config):
        super().__init__()
        self.attention = BertAttention(bert_config)
        self.intermediate = BertIntermediate(bert_config)
        self.output = BertOutput(bert_config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        # outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        # outputs = (layer_output,) + outputs
        return layer_output
