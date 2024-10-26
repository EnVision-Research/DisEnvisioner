import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class EVAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for EnVisioner for PyTorch 2.0.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale_object=1.0, scale_others=1.0, num_tokens_object=4, num_tokens_others=4):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("EVAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale_object = scale_object
        self.scale_others = scale_others
        self.num_tokens_object = num_tokens_object
        self.num_tokens_others = num_tokens_others

        self.to_k_object = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_k_others = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_object = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_others = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)


    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # get encoder_hidden_states, encoder_hidden_states_object, encoder_hidden_states_others
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            end_pos = encoder_hidden_states.shape[1] - (self.num_tokens_object + self.num_tokens_others)
            encoder_hidden_states, encoder_hidden_states_object, encoder_hidden_states_others = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:end_pos+self.num_tokens_object, :],
                encoder_hidden_states[:, end_pos+self.num_tokens_object:end_pos+self.num_tokens_object+self.num_tokens_others, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # ------------------- cross attention for textual prompt -------------------
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # ------------------- cross attention for image prompt (object) -------------------
        key_object = self.to_k_object(encoder_hidden_states_object)
        value_object = self.to_v_object(encoder_hidden_states_object)

        key_object = key_object.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_object = value_object.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states_object = F.scaled_dot_product_attention(
            query, key_object, value_object, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        hidden_states_object = hidden_states_object.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_object = hidden_states_object.to(query.dtype)

        # ------------------- cross attention for image prompt (others) -------------------
        key_others = self.to_k_others(encoder_hidden_states_others)
        key_others = key_others.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        value_others = self.to_v_others(encoder_hidden_states_others)
        value_others = value_others.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states_others = F.scaled_dot_product_attention(
            query, key_others, value_others, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        
        hidden_states_others = hidden_states_others.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_others = hidden_states_others.to(query.dtype)

        # with torch.no_grad():
        #     self.attn_map_object = query @ key_object.transpose(-2, -1).softmax(dim=-1)
        #     self.attn_map_others = query @ key_others.transpose(-2, -1).softmax(dim=-1)

        # ------------------- merge three cross attention -------------------
        hidden_states = hidden_states + \
            self.scale_object * hidden_states_object + \
            self.scale_others * hidden_states_others

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class EVAttnProcessor2_0_W_IP(torch.nn.Module):
    r"""
    Attention processor for EnVisioner for PyTorch 2.0.
    """

    def __init__(self, 
                 hidden_size, cross_attention_dim=None, 
                 scale_object=1.0, scale_others=1.0, scale_ip=1.0, 
                 num_tokens_object=4, num_tokens_others=4, num_tokens_ip=16):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("EVAttnProcessor2_0_W_IP requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        self.scale_object = scale_object
        self.scale_others = scale_others
        self.scale_ip = scale_ip

        self.num_tokens_object = num_tokens_object
        self.num_tokens_others = num_tokens_others
        self.num_tokens_ip = num_tokens_ip

        self.to_k_object = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_k_others = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

        self.to_v_object = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_others = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        
        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)


    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # get encoder_hidden_states, encoder_hidden_states_object, encoder_hidden_states_others
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            end_pos = encoder_hidden_states.shape[1] - (self.num_tokens_object + self.num_tokens_others + self.num_tokens_ip)
            encoder_hidden_states, encoder_hidden_states_object, encoder_hidden_states_others, encoder_hidden_states_ip = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:end_pos+self.num_tokens_object, :],
                encoder_hidden_states[:, end_pos+self.num_tokens_object:end_pos+self.num_tokens_object+self.num_tokens_others, :],
                encoder_hidden_states[:, end_pos+self.num_tokens_object+self.num_tokens_others:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # ------------------- cross attention for textual prompt -------------------
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # ------------------- cross attention for image prompt (object) -------------------
        key_object = self.to_k_object(encoder_hidden_states_object)
        value_object = self.to_v_object(encoder_hidden_states_object)

        key_object = key_object.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_object = value_object.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states_object = F.scaled_dot_product_attention(
            query, key_object, value_object, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        hidden_states_object = hidden_states_object.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_object = hidden_states_object.to(query.dtype)

        # ------------------- cross attention for image prompt (others) -------------------
        key_others = self.to_k_others(encoder_hidden_states_others)
        key_others = key_others.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        value_others = self.to_v_others(encoder_hidden_states_others)
        value_others = value_others.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states_others = F.scaled_dot_product_attention(
            query, key_others, value_others, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        
        hidden_states_others = hidden_states_others.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_others = hidden_states_others.to(query.dtype)

        with torch.no_grad():
            self.attn_map_object = query @ key_object.transpose(-2, -1).softmax(dim=-1)
            self.attn_map_others = query @ key_others.transpose(-2, -1).softmax(dim=-1)

        # ------------------- cross attention for image prompt (ip) -------------------
        key_ip = self.to_k_ip(encoder_hidden_states_ip)
        key_ip = key_ip.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        value_ip = self.to_v_ip(encoder_hidden_states_ip)
        value_ip = value_ip.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states_ip = F.scaled_dot_product_attention(
            query, key_ip, value_ip, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        hidden_states_ip = hidden_states_ip.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_ip = hidden_states_ip.to(query.dtype)

        # ------------------- merge cross attentions -------------------
        hidden_states = hidden_states + \
            self.scale_object * hidden_states_object + \
            self.scale_others * hidden_states_others + \
            self.scale_ip * hidden_states_ip

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
