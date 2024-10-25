import torch

class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class Projector(torch.nn.Module):
    def __init__(self, cross_attention_dim=1024, input_embedding_dim=768, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        self.proj_in = torch.nn.Linear(input_embedding_dim, clip_embeddings_dim)
        torch.nn.init.constant_(self.proj_in.weight, 1)
        torch.nn.init.constant_(self.proj_in.bias, 0)
        
        self.image_proj_model_object = ImageProjModel( # TODO
            cross_attention_dim=cross_attention_dim,
            clip_embeddings_dim=clip_embeddings_dim,
            clip_extra_context_tokens=clip_extra_context_tokens,
        )
        self.image_proj_model_others = ImageProjModel( # TODO
            cross_attention_dim=cross_attention_dim,
            clip_embeddings_dim=clip_embeddings_dim,
            clip_extra_context_tokens=clip_extra_context_tokens,
        )

    def forward(self, image_embeds):
        image_embeds_object = self.image_proj_model_object(self.proj_in(image_embeds[:,0]))
        image_embeds_others = self.image_proj_model_others(self.proj_in(image_embeds[:,1]))
        return image_embeds_object, image_embeds_others

class EnVisioner(torch.nn.Module):
    def __init__(self, image_proj_model, adapter_modules):
        super().__init__()
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

class EnVisioner_IP(torch.nn.Module):
    def __init__(self, image_proj_model, image_proj_model_ip, adapter_modules):
        super().__init__()
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        self.image_proj_model_ip = image_proj_model_ip