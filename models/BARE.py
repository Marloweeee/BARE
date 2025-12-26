import torch
import torch.nn as nn
import torch.nn.functional as F
from .vl_transformer import build_vl_transformer
from torchvision.transforms import Resize
from .vl_fusion import R2E
from peft import get_peft_model, LoraConfig
import math
from .beit.modeling_utils import BEiT3Wrapper, _get_base_config, _get_large_config
from .beit.utils import load_state_dict
import numpy as np


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LinguisticSalienceModulator(nn.Module):
    """
    Linguistic Salience Modulator (LSM)

    Performs debiasing on input text_tensors (batchsize × N_t × d):
    - r_i: referential importance, high values indicate the token is important for grounding
    - b_i: bias degree, high values indicate the token is a template/bias word
    - v_bias: learnable bias direction vector

    Formula:
        h_i = LN(W_h @ e_i)
        r_i = sigmoid(w_r^T @ h_i)
        b_i = sigmoid(w_b^T @ h_i)
        e_i^deb = r_i * e_i - lambda * b_i * v_bias
        e_i_out = e_i + alpha * (e_i^deb - e_i)
    """

    def __init__(self, embed_dim, hidden_dim=None, lambda_debias=0.1, alpha=0.5, learnable_alpha=True):
        """
        Args:
            embed_dim: input embedding dimension d
            hidden_dim: hidden layer dimension, defaults to embed_dim
            lambda_debias: debiasing intensity coefficient
            alpha: residual connection coefficient (between 0-1)
            learnable_alpha: whether to make alpha learnable
        """
        super().__init__()

        pass

    def get_alpha(self, ref_tensor=None):
        """
        Get alpha value, optionally align dtype/device

        Args:
            ref_tensor: reference tensor used to align dtype and device (e.g., mixed precision scenarios)
        """
        pass

    @property
    def alpha(self):
        """Get alpha value (not aligned with dtype, compatible with old code)"""
        pass

    def get_normalized_v_bias(self):
        """Get L2 normalized bias vector"""
        pass

    def forward(self, text_tensors, text_mask=None):
        """
        Args:
            text_tensors: (batch_size, N_t, d) text embeddings
            text_mask: (batch_size, N_t) text mask, True indicates padding positions

        Returns:
            debiased_tensors: (batch_size, N_t, d) debiased text embeddings
            gate_info: dict containing intermediate information such as r_i, b_i, etc., for potential auxiliary losses
        """
        pass

    def get_v_bias_reg_loss(self):
        """Get L2 regularization loss for v_bias to prevent norm explosion"""
        pass


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TOKEN_MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.activation_fn = QuickGELU()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class BARE(BEiT3Wrapper):

    def __init__(
        self,
        args,
        freeze_layer=-1,
        vision_embed_proj_interpolate=False,
        pretrain=None,
    ):
        if pretrain is None:
            # If --beit_model is specified in the command line, prioritize that path
            if hasattr(args, "beit_model") and args.beit_model:
                pretrain = args.beit_model
            else:
                if args.vit_type == "base":
                    pretrain = "./weights/beit3_base_indomain_patch16_224.pth"
                elif args.vit_type == "large":
                    # Corresponding to large version weights (already in the repository's weights directory)
                    pretrain = "./weights/beit3_large_indomain_patch16_224.pth"
                else:
                    raise ValueError(f"Unsupported ViT type: {args.vit_type}")
        if args.vit_type == "base":
            beit_args = _get_base_config(
                img_size=args.imsize,
                patch_size=args.patch_size,
                drop_path_rate=args.drop_path_rate,
                vocab_size=args.vocab_size,
            )
        elif args.vit_type == "large":
            beit_args = _get_large_config(
                img_size=args.imsize,
                patch_size=args.patch_size,
                rop_path_rate=args.drop_path_rate,
                vocab_size=args.vocab_size,
            )
        else:
            raise TypeError(
                "please select the <vit_type> from ['base','large']")

        super(BARE, self).__init__(args=beit_args)

        self.embed_dim = beit_args.encoder_embed_dim
        self.hidden_size = beit_args.encoder_ffn_embed_dim
        self.reg_token = nn.Embedding(1, self.embed_dim)
        self.visu_tokens = int((args.imsize/args.patch_size)**2)
        self.text_tokens = args.max_query_len
        self.totle_tokens = self.visu_tokens+self.text_tokens+1+1
        self.vl_pos_embed = nn.Embedding(self.totle_tokens, self.embed_dim)

        self.visu_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.text_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.visu_aware_feature_interaction = R2E(
            d_model=self.embed_dim, nhead=8, dropout=0.0,
            activation='relu', normalize_before=False)

        self.vl_transformer = build_vl_transformer(args=args)
        self.vision_embed_proj_interpolate = vision_embed_proj_interpolate
        # reg
        self.bbox_embed = MLP(self.embed_dim, self.embed_dim, 4, 3)

        # visual token align
        self.visu_token_norm = nn.LayerNorm(self.embed_dim, eps=1e-05)
        self.visu_token_mlp = TOKEN_MLP(self.embed_dim, 3072)

        # seg
        self.seg_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=(2, 2), stride=(2, 2),
                               padding=(0, 0), output_padding=(0, 0), bias=False),  # bias=False
            nn.ConvTranspose2d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=(2, 2), stride=(2, 2),
                               padding=(0, 0), output_padding=(0, 0), bias=False),  # bias=False
            nn.ConvTranspose2d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=(2, 2), stride=(2, 2),
                               padding=(0, 0), output_padding=(0, 0), bias=False)  # bias=False
        )

        # Linguistic Salience Modulator (LSM)
        # Get debiasing parameters from args, use defaults if not present
        lambda_debias = getattr(args, 'lambda_debias', 0.1)
        alpha_debias = getattr(args, 'alpha_debias', 0.5)
        learnable_alpha = getattr(args, 'learnable_alpha', True)
        self.lsm = LinguisticSalienceModulator(
            embed_dim=self.embed_dim,
            hidden_dim=self.embed_dim,
            lambda_debias=lambda_debias,
            alpha=alpha_debias,
            learnable_alpha=learnable_alpha
        )

        # load pretrain checkpoint
        if isinstance(pretrain, str):
            self.load_model_and_may_interpolate(pretrain)
        # freeze the encoder
        if freeze_layer >= 0:
            self.frozen_stages = freeze_layer if freeze_layer <= len(
                self.beit3.encoder.layers) else len(self.beit3.encoder.layers)
            self._freeze_stages()

        self.set_BARE(args)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:

            for i in range(1, self.frozen_stages + 1):
                m = self.beit3.encoder.layers[i - 1]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def set_BARE(self, args):
        open_lora_parameter_update = True
        close_self_attn_parameter_update = False
        close_multiway_experts_parameter_update = False

        if open_lora_parameter_update:
            # target modules for lora
            target_modules = ["q_proj.A", "k_proj.A", "v_proj.A", "out_proj.A",
                              "q_proj.B", "k_proj.B", "v_proj.B", "out_proj.B",
                              ]
            # construct lora config
            peft_config = LoraConfig(target_modules=target_modules,
                                     inference_mode=False,
                                     r=32, lora_alpha=16,
                                     lora_dropout=0.1, bias='none')

            self.beit3 = get_peft_model(self.beit3, peft_config)
            self.beit3.print_trainable_parameters()
            for parameter in self.beit3.parameters():
                parameter.requires_grad_(False)
            self.beit3.print_trainable_parameters()

            if args.vit_type == "base":
                print("Open LORA parameter update in stage 3")
                self.beit3 = get_peft_model(self.beit3, peft_config)
                for parameter in self.beit3.parameters():
                    parameter.requires_grad_(False)
                for name, parm in self.beit3.encoder.layers.named_parameters():
                    if "lora_A" in str(name).split(".") or "lora_B" in str(name).split("."):
                        if "0" in str(name).split(".") or "1" in str(name).split(".") \
                                or "2" in str(name).split(".") or "3" in str(name).split(".") \
                                or "4" in str(name).split(".") or "5" in str(name).split(".") \
                                or "6" in str(name).split(".") or "7" in str(name).split(".") \
                                or "8" in str(name).split(".") or "9" in str(name).split(".") \
                                or "10" in str(name).split(".") or "11" in str(name).split("."):
                            print("parameter name:", name)
                            parm.requires_grad_(True)
                    else:
                        parm.requires_grad_(False)
                self.beit3.print_trainable_parameters()

            elif args.vit_type == "large":
                print("Open LORA parameter update in stage 4")
                self.beit3 = get_peft_model(self.beit3, peft_config)
                for parameter in self.beit3.parameters():
                    parameter.requires_grad_(False)
                for name, parm in self.beit3.encoder.layers.named_parameters():
                    if "lora_A" in str(name).split(".") or "lora_B" in str(name).split("."):
                        if "0" in str(name).split(".") or "1" in str(name).split(".") \
                                or "2" in str(name).split(".") or "3" in str(name).split(".") \
                                or "4" in str(name).split(".") or "5" in str(name).split(".") \
                                or "6" in str(name).split(".") or "7" in str(name).split(".") \
                                or "8" in str(name).split(".") or "9" in str(name).split(".") \
                                or "10" in str(name).split(".") or "11" in str(name).split(".") \
                                or "12" in str(name).split(".") or "13" in str(name).split(".") \
                                or "14" in str(name).split(".") or "15" in str(name).split(".") \
                                or "16" in str(name).split(".") or "17" in str(name).split(".") \
                                or "18" in str(name).split(".") or "19" in str(name).split(".") \
                                or "20" in str(name).split(".") or "21" in str(name).split(".") \
                                or "22" in str(name).split(".") or "23" in str(name).split("."):
                            print("parameter name:", name)
                            parm.requires_grad_(True)
                    else:
                        parm.requires_grad_(False)
                self.beit3.print_trainable_parameters()

    def load_model_and_may_interpolate(self, ckpt_path, model_key="model|module", model_prefix=""):
        if ckpt_path.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                ckpt_path, map_location="cpu", check_hash=True)
        else:
            checkpoint = torch.load(ckpt_path, map_location="cpu")

        print("Load ckpt from %s" % ckpt_path)
        checkpoint_model = None
        for model_key in model_key.split("|"):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break

        if checkpoint_model is None:
            checkpoint_model = checkpoint

        state_dict = self.state_dict()
        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        for pos_embed_key in (
            "vision_pos_embed",
            "pos_embed",
            "beit3.encoder.embed_positions.A.weight",
        ):
            if pos_embed_key in checkpoint_model:
                pos_embed_checkpoint = checkpoint_model[pos_embed_key]
                embedding_size = pos_embed_checkpoint.shape[-1]
                if pos_embed_key == "beit3.encoder.embed_positions.A.weight":
                    # being consistent with Fairseq, which starts from 2 for position embedding
                    torchscale_model = True
                    num_patches = self.beit3.vision_embed.num_patches
                    num_extra_tokens = self.beit3.vision_embed.num_position_embeddings() + 2 - \
                        num_patches
                else:
                    torchscale_model = False
                    num_patches = self.patch_embed.num_patches
                    num_extra_tokens = getattr(
                        self, pos_embed_key).shape[-2] - num_patches
                # height (== width) for the checkpoint position embedding
                orig_size = int(
                    (pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int(num_patches**0.5)
                # class_token and dist_token are kept unchanged
                if orig_size != new_size:
                    print("Position interpolate from %dx%d to %dx%d" %
                          (orig_size, orig_size, new_size, new_size))
                    if torchscale_model:
                        extra_tokens = pos_embed_checkpoint[:num_extra_tokens].unsqueeze(
                            0)
                        # only the position tokens are interpolated
                        pos_tokens = pos_embed_checkpoint[num_extra_tokens:]
                    else:
                        extra_tokens = pos_embed_checkpoint[:,
                                                            :num_extra_tokens]
                        # only the position tokens are interpolated
                        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    pos_tokens = pos_tokens.reshape(
                        -1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2).float()
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens,
                        size=(new_size, new_size),
                        mode="bicubic",
                        align_corners=False,
                    )
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                    new_pos_embed = torch.cat(
                        (extra_tokens, pos_tokens), dim=1)
                    if torchscale_model:
                        new_pos_embed = new_pos_embed.squeeze(0)
                    checkpoint_model[pos_embed_key] = new_pos_embed

        if (
            checkpoint_model["beit3.vision_embed.proj.weight"].shape != self.beit3.vision_embed.proj.weight.shape
        ) and self.vision_embed_proj_interpolate:
            vision_embed_proj_weight = checkpoint_model["beit3.vision_embed.proj.weight"]
            new_size = self.beit3.vision_embed.proj.weight.shape[-2:]
            vision_embed_proj_weight = torch.nn.functional.interpolate(
                vision_embed_proj_weight.float(),
                size=new_size,
                mode="bicubic",
                align_corners=False,
            )
            checkpoint_model["beit3.vision_embed.proj.weight"] = vision_embed_proj_weight

        load_state_dict(self, checkpoint_model, prefix=model_prefix)

    def get_debias_reg_loss(self, weight=1e-4):
        """
        Get regularization loss for the debiasing module

        Args:
            weight: regularization weight

        Returns:
            L2 regularization loss for v_bias, used to prevent bias vector norm explosion
        """
        return weight * self.lsm.get_v_bias_reg_loss()

    def get_last_gate_info(self):
        """
        Get the gating information from the last forward pass

        Returns:
            dict containing:
                - r_i: (B, N_t) referential importance of each token
                - b_i: (B, N_t) bias degree of each token
                - v_bias: (d,) learned bias direction vector
                - alpha: residual connection coefficient
        """
        if hasattr(self, '_last_gate_info'):
            return self._last_gate_info
        return None


    def get_masks(self, img_mask, text_mask):
        torch_resize = Resize([14, 14])
        visu_mask = torch_resize(img_mask)
        visu_mask = visu_mask.to(torch.bool)
        visu_mask = visu_mask.flatten(1)

        text_mask = text_mask.to(torch.bool)
        text_mask = ~text_mask

        return visu_mask, text_mask

    def forward(self, img_data, text_data):
        batch_size = img_data.tensors.shape[0]
        image_tensors = img_data.tensors
        text_tensors, text_mask = text_data.tensors, text_data.mask

        # ========== Text debiasing gating process (before encoder) ==========
        # Get the underlying beit3 model (handles PEFT wrapper)
        beit3_base = self.beit3.model if hasattr(
            self.beit3, 'model') else self.beit3

        # 1. Get text embeddings
        text_embed = beit3_base.text_embed(text_tensors)  # (B, N_t, d)

        # 2. Debiasing text embeddings via LSM
        text_embed_debiased, gate_info = self.lsm(
            text_embed, text_mask)
        self._last_gate_info = gate_info

        # 3. Get vision embeddings
        vision_embed = beit3_base.vision_embed(
            image_tensors, None)  # (B, N_v+1, d)
        multiway_split_position = vision_embed.size(1)

        # 4. Concat and construct mask
        x = torch.cat([vision_embed, text_embed_debiased], dim=1)
        if text_mask is not None:
            encoder_padding_mask = torch.cat([
                torch.zeros(vision_embed.shape[:-1],
                            device=vision_embed.device).bool(),
                text_mask.bool() if text_mask.dtype != torch.bool else text_mask,
            ], dim=1)
        else:
            encoder_padding_mask = None

        # 5. Call encoder
        outputs = beit3_base.encoder(
            src_tokens=None,
            encoder_padding_mask=encoder_padding_mask,
            token_embeddings=x,
            multiway_split_position=multiway_split_position,
        )
        # ========================================

        x = outputs["encoder_out"]

        cls_feat, img_feat, text_feat = x[:, 0], x[:, 1: -
                                                   text_tensors.shape[-1]], x[:, -text_tensors.shape[-1]:]

        eos_token = text_feat[:, 0, :]
        visu_mask, text_mask = self.get_masks(img_data.mask, text_mask)
        reg_src = self.reg_token.weight.unsqueeze(0).repeat(
            img_data.tensors.shape[0], 1, 1)  # B * 1 * hidden_dim

        cls_src = self.visu_proj(cls_feat.float())  # B * 768
        visu_src = self.visu_proj(img_feat.float())  # B * 196 * 768
        eos_src = self.text_proj(eos_token.float())  # B * 768
        text_src = self.text_proj(text_feat.float())  # B * 77 * 768

        reg_src = reg_src.permute(1, 0, 2)  # 1 * B * 768
        cls_src = cls_src.unsqueeze(0)  # 1 * B * 768
        visu_src = visu_src.permute(1, 0, 2)  # 196 * B * 768
        text_src = text_src.permute(1, 0, 2)  # 77 * B * 768

        # mask
        reg_mask = torch.zeros((batch_size, 1)).to(
            reg_src.device).to(torch.bool)
        cls_mask = torch.zeros((batch_size, 1)).to(
            reg_src.device).to(torch.bool)

        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        vl_mask = torch.cat([reg_mask, cls_mask, visu_mask, text_mask], dim=1)

        # # refine visu and text features
        text_refine_src = self.visu_aware_feature_interaction(
            visu_src, text_src, visu_mask, text_mask, vl_pos)

        vl_src = torch.cat(
            [reg_src, cls_src, visu_src, text_refine_src], dim=0)
        # vl_src = torch.cat([reg_src,cls_src, visu_src, text_src], dim=0)
        vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos)  # (1+L+N)xBxC

        # reg
        box_hs = vg_hs[0]
        pred_box = self.bbox_embed(box_hs).sigmoid()

        # logits
        img_cls_embed = cls_src/cls_src.norm(p=2, dim=-1, keepdim=True)
        txt_eos_embed = eos_src/eos_src.norm(p=2, dim=-1, keepdim=True)
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        logits_per_image = logit_scale * \
            torch.matmul(img_cls_embed.squeeze(), txt_eos_embed.t())

        # visu and text features
        vg_hs_visu_features = vg_hs[2: 2 +
                                    self.visu_tokens].permute(1, 0, 2)  # B, N, C
        vg_hs_text_features = vg_hs[2 +
                                    self.visu_tokens:].permute(1, 0, 2)  # B, L, C
        vg_hs_text_eos_embed = vg_hs_text_features[torch.arange(
            vg_hs_text_features.shape[0]), text_tensors.argmax(dim=-1)]
        vg_hs_text_eos_embed = vg_hs_text_eos_embed / \
            vg_hs_text_eos_embed.norm(p=2, dim=-1, keepdim=True)

        # each visual token's similarity to the text eos token
        visu_last_layer_feat = self.visu_token_mlp(
            self.visu_token_norm(vg_hs_visu_features))  # B,196,768
        visu_token_similarity = torch.mul(
            vg_hs_text_eos_embed.unsqueeze(1).repeat(
                1, self.visu_tokens, 1),  # B, 196, 768
            visu_last_layer_feat  # B, 196, 768
        )  # B, 196, 768
        visu_token_similarity = visu_token_similarity.sum(
            axis=-1, keepdim=False)  # torch.Size([B, 196])

        # seg
        patch_num = int(math.sqrt(vg_hs_visu_features.shape[1]))
        channels = vg_hs_visu_features.shape[-1]
        assert patch_num**2 == vg_hs_visu_features.shape[1]
        seg_features = vg_hs_visu_features.permute(0, 2, 1).reshape(
            batch_size, channels, patch_num, patch_num)
        seg_features = self.seg_conv(seg_features)  # B, C, 112, 112
        seg_features = seg_features.permute(0, 2, 3, 1)
        seg_mask = torch.mul(vg_hs_text_eos_embed.reshape(batch_size, 1, 1, vg_hs_text_eos_embed.shape[-1]).repeat(1, seg_features.shape[1], seg_features.shape[2], 1),
                             seg_features)
        seg_mask = seg_mask.sum(axis=-1, keepdim=False).unsqueeze(1)  # B 1 H W
        # print(seg_mask.shape)

        return pred_box, seg_mask, logits_per_image, visu_token_similarity

    