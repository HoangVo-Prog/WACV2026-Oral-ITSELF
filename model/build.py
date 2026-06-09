import copy
from model import objectives
from .clip_model import Transformer, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights,tokenize
import torch
import torch.nn as nn
from .grab import TexualEmbeddingLayer, VisualEmbeddingLayer
from .prototype import PrototypeBranch
from torch.cuda.amp import autocast


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # ipdb.set_trace()
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        outputs = self.transformer([x])
        x = outputs[0]
        att = outputs[1]
        x = x.permute(1, 0, 2)  # LND -> NLD   # x,att
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        text_feature = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return text_feature



class ITSELF(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        self.grab_embed_dim = 4096
        self.args = args
        self.train_num_classes = num_classes
        self.prototype_enabled = (
            getattr(args, "prototype", False)
            or getattr(args, "use_loss_id", False)
        )
        prototype_feature = getattr(args, "prototype_feature", "auto")
        use_proto_local = self.prototype_enabled and not args.only_global and prototype_feature in ("auto", "local")
        if 'cid' in self.current_task:
            self.num_classes = num_classes + 1
            self.classifier_global = nn.Linear(self.embed_dim , self.num_classes)
            nn.init.normal_(self.classifier_global.weight.data, std=0.001)
            nn.init.constant_(self.classifier_global.bias.data, val=0.0)
            self.mlp_global = nn.Sequential(nn.Linear(2 * self.embed_dim, self.embed_dim),nn.LayerNorm(self.embed_dim),nn.GELU())
            self.classifier_id_global = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier_id_global.weight.data, std=0.001)
            nn.init.constant_(self.classifier_id_global.bias.data, val=0.0)
            if not args.only_global:
                self.classifier_grab = nn.Linear(self.grab_embed_dim, self.num_classes)
                nn.init.normal_(self.classifier_grab.weight.data, std=0.001)
                nn.init.constant_(self.classifier_grab.bias.data, val=0.0)
                self.mlp_grab = nn.Sequential(nn.Linear(2 * self.grab_embed_dim, self.grab_embed_dim),nn.LayerNorm(self.grab_embed_dim),nn.GELU())
                self.classifier_id_grab = nn.Linear(self.grab_embed_dim, self.num_classes)
                nn.init.normal_(self.classifier_id_grab.weight.data, std=0.001)
                nn.init.constant_(self.classifier_id_grab.bias.data, val=0.0)
                self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
                self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)

        if not args.only_global and not hasattr(self, "visul_emb_layer") and ('tal' in self.current_task or use_proto_local):
            self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
            self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)

        if self.prototype_enabled:
            prototype_feature_dim = self.grab_embed_dim if use_proto_local else self.embed_dim
            self.prototype_branch = PrototypeBranch(args, num_classes, prototype_feature_dim)
        else:
            self.prototype_branch = None
                
        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
  
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+') if l.strip() and l.strip() != 'proto']
        print(f'Training Model with {self.current_task} tasks')
    
    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()
      
    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text.long())
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_grab(self, image):
        x,atten_i = self.base_model.encode_image(image)
        i_grab_f = self.visul_emb_layer(x, atten_i)
        return i_grab_f.float()

    def encode_text_grab(self, text):
        x,atten_t = self.base_model.encode_text(text.long())
        t_grab_f = self.texual_emb_layer(x, text, atten_t)
        return t_grab_f.float()
    
    def rollout(self, attentions: torch.Tensor, 
                head_fusion = 'mean', 
                discard: bool = True,
                discard_ratios: list = [0.25, 1., 1., 1., 0.25, 0.25, 1., 1., 1., 1., 0.25, 0.25], 
                start_layer: int = 4, 
                skip_layer: list = [5,6,7,8,9,10]):
        
        if len(attentions.shape) == 5:
            L, B, _, N, _ = attentions.shape
        else:
            L, B, N, _ = attentions.shape
        device = attentions.device
        result = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)  # [B, N, N]
                    
        for layer in range(start_layer, L):
            if layer in skip_layer:
                continue
            attn = attentions[layer]  
            # have H shape (L, B, H, N, N)
            if len(attentions.shape) == 5:
                with torch.no_grad():
                    if head_fusion == "mean":
                        attn = attn.mean(axis=1) # [B, H, N, N] --> axis == 1
                    elif head_fusion == "max":
                        attn = attn.max(axis=1)[0]
                    elif head_fusion == "min":
                        attn = attn.min(axis=1)[0]
                    else:
                        raise "Attention head fusion type Not supported"
            
            if discard:
                discard_ratio = discard_ratios[layer]
                flat = attn.view(B, -1)  # [B, N*N]
                num_to_discard = int(flat.size(-1) * discard_ratio)

                if num_to_discard > 0:
                    _, indices = flat.topk(num_to_discard, dim=-1, largest=False)
                    for b in range(B):
                        idx = indices[b]
                        idx = idx[idx != 0]
                        flat[b, idx] = 0
                    attn = flat.view(B, N, N)

            I = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
            attn = (attn + I) / 2.0
            attn = attn / attn.sum(dim=-1, keepdim=True)
            result = torch.bmm(attn, result)

        return result  # [B, N, N]

    def _compute_host_embeddings(self, images, caption_ids, current_step=None):
        if self.args.return_all:
            image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids, return_all=True, average_attn_weights = self.args.average_attn_weights)
            i_feats = image_feats[:, 0, :].float()
            t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
            if self.args.topk_type == 'mean':
                atten_i = torch.stack(atten_i, dim=0)
                atten_t = torch.stack(atten_t, dim=0)
                atten_i = atten_i.mean(0)
                atten_t = atten_t.mean(0)
                if current_step is not None:
                    i_grab_f = self.visul_emb_layer(image_feats, atten_i, current_step)
                    t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t, current_step)
                else:
                    i_grab_f = self.visul_emb_layer(image_feats, atten_i)
                    t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            elif self.args.topk_type == 'std':
                atten_i = torch.stack(atten_i, dim=0)
                atten_t = torch.stack(atten_t, dim=0)
                atten_i = atten_i.std(0, unbiased=False)
                atten_t = atten_t.std(0, unbiased=False)
                if current_step is not None:
                    i_grab_f = self.visul_emb_layer(image_feats, atten_i, current_step)
                    t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t, current_step)
                else:
                    i_grab_f = self.visul_emb_layer(image_feats, atten_i)
                    t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            elif self.args.topk_type == 'layer_index' and self.args.layer_index is not None:
                atten_i = atten_i[self.args.layer_index]
                atten_t = atten_t[self.args.layer_index]
                if current_step is not None:
                    i_grab_f = self.visul_emb_layer(image_feats, atten_i, current_step)
                    t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t, current_step)
                else:
                    i_grab_f = self.visul_emb_layer(image_feats, atten_i)
                    t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            elif self.args.topk_type == 'custom':
                atten_i = torch.stack(atten_i, dim=0)
                atten_t = torch.stack(atten_t, dim=0)
                atten_i = self.rollout(atten_i)
                atten_t = self.rollout(atten_t)
                if not self.args.only_global:
                    if current_step is not None:
                        i_grab_f = self.visul_emb_layer(image_feats, atten_i, current_step)
                        t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t, current_step)
                    else:
                        i_grab_f = self.visul_emb_layer(image_feats, atten_i)
                        t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            else:
                if not self.args.only_global:
                    i_grab_f = self.visul_emb_layer(image_feats, atten_i)
                    t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
        else:
            image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
            i_feats = image_feats[:, 0, :].float()
            t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
            if not self.args.only_global:
                i_grab_f = self.visul_emb_layer(image_feats, atten_i)
                t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)

        features = {"i_feats": i_feats, "t_feats": t_feats}
        if not self.args.only_global:
            features.update({"i_grab_f": i_grab_f.float(), "t_grab_f": t_grab_f.float()})
        return features

    def _select_prototype_features(self, features):
        if self.prototype_branch is not None and self.prototype_branch.use_local and not self.args.only_global:
            return features["i_grab_f"], features["t_grab_f"]
        return features["i_feats"], features["t_feats"]

    def _diagnostic_layer_attention(self, attentions):
        head_averaged = [a.mean(dim=1) if a.ndim == 4 else a for a in attentions]
        if not getattr(self.args, "return_all", False):
            return head_averaged[-1], "last_layer"

        topk_type = getattr(self.args, "topk_type", "mean")
        stacked = torch.stack(head_averaged, dim=0)
        if topk_type == "mean":
            return stacked.mean(0), "mean_layer"
        if topk_type == "std":
            return stacked.std(0, unbiased=False), "std_layer"
        if topk_type == "layer_index" and getattr(self.args, "layer_index", None) is not None:
            idx = int(getattr(self.args, "layer_index"))
            idx = max(-len(head_averaged), min(len(head_averaged) - 1, idx))
            return head_averaged[idx], f"layer_{idx}"
        if topk_type == "custom":
            return self.rollout(stacked), "rollout"
        return head_averaged[-1], "last_layer"

    @staticmethod
    def _diagnostic_topk_count(total_tokens, special_tokens, ratio, current_step=None):
        usable = max(int(total_tokens) - int(special_tokens), 1)
        if current_step is not None:
            ratio_start = 0.65
            ratio_end = 0.5
            total_steps = 10 * 145
            current_step = min(max(int(current_step), 1), total_steps)
            progress = current_step / total_steps
            k = int(usable * ratio_start * ((ratio_end / ratio_start) ** progress))
        else:
            k = int(usable * float(ratio))
        return max(1, min(usable, k))

    def _visual_evidence_prior(self, attention, use_local, current_step=None):
        bsz, ntokens, _ = attention.shape
        scores = attention[:, 0, :].float().clone()
        scores[:, 0] = 0.0
        prior = scores[:, 1:].clamp_min(0.0)
        mask = torch.ones((bsz, ntokens - 1), device=attention.device, dtype=torch.bool)
        if use_local:
            ratio = getattr(getattr(self, "visul_emb_layer", None), "ratio", getattr(self.args, "select_ratio", 0.4))
            k = self._diagnostic_topk_count(ntokens, 1, ratio, current_step=current_step)
            top_idx = scores.topk(dim=-1, k=k).indices
            full_mask = torch.zeros_like(scores, dtype=torch.bool)
            full_mask.scatter_(1, top_idx, True)
            mask = full_mask[:, 1:]
            prior = prior * mask.to(dtype=prior.dtype)
        return prior, mask

    def _text_evidence_prior(self, attention, caption_ids, use_local, current_step=None):
        bsz, ntokens, _ = attention.shape
        text_mask = caption_ids.ne(0)
        eot = caption_ids.argmax(dim=-1)
        rows = torch.arange(bsz, device=attention.device)
        scores = attention[rows, eot, :].float().clone()
        scores[:, 0] = 0.0
        scores[rows, eot] = 0.0
        token_mask = text_mask.bool().clone()
        token_mask[:, 0] = False
        token_mask[rows, eot] = False
        prior = scores.clamp_min(0.0) * token_mask.to(dtype=scores.dtype)
        if use_local:
            ratio = getattr(getattr(self, "texual_emb_layer", None), "ratio", getattr(self.args, "select_ratio", 0.4))
            k = self._diagnostic_topk_count(ntokens, 2, ratio, current_step=current_step)
            masked_scores = scores.masked_fill(~token_mask, float("-inf"))
            top_idx = masked_scores.topk(dim=-1, k=k).indices
            top_mask = torch.zeros_like(token_mask)
            top_mask.scatter_(1, top_idx, True)
            token_mask = token_mask & top_mask
            prior = prior * token_mask.to(dtype=prior.dtype)
        return prior, token_mask

    def _visual_tokens_for_prototype_evidence(self, image_tokens):
        patch_tokens = image_tokens[:, 1:, :].float()
        branch = getattr(self, "prototype_branch", None)
        if branch is not None and branch.use_local and not self.args.only_global:
            layer = self.visul_emb_layer
            patch_tokens = l2norm(patch_tokens, dim=-1)
            dtype = layer.fc.weight.dtype
            patch_tokens = patch_tokens.to(dtype=dtype)
            return (layer.mlp(patch_tokens) + layer.fc(patch_tokens)).float()
        return patch_tokens

    def _text_tokens_for_prototype_evidence(self, text_tokens):
        token_features = text_tokens.float()
        branch = getattr(self, "prototype_branch", None)
        if branch is not None and branch.use_local and not self.args.only_global:
            layer = self.texual_emb_layer
            token_features = l2norm(token_features, dim=-1)
            dtype = layer.linear.weight.dtype
            token_features = token_features.to(dtype=dtype)
            return (layer.mlp(token_features) + layer.linear(token_features)).float()
        return token_features

    @torch.no_grad()
    def collect_prototype_evidence(
        self,
        batch,
        current_step=None,
        max_prototypes_per_id=None,
        include_raw_heads=False,
    ):
        branch = getattr(self, "prototype_branch", None)
        if branch is None:
            raise RuntimeError("Prototype evidence requested, but the model has no prototype branch.")
        if not branch.is_ready():
            raise RuntimeError("Prototype evidence requested, but the prototype bank is not initialized/loaded.")

        images = batch["images"]
        caption_ids = batch["caption_ids"].long()
        pids = batch["pids"].long()
        image_tokens, image_attentions = self.base_model.encode_image_all_atten(
            images,
            average_attn_weights=not include_raw_heads,
        )
        text_tokens, text_attentions = self.base_model.encode_text_all_atten(
            caption_ids,
            average_attn_weights=not include_raw_heads,
        )
        image_attention, image_attention_source = self._diagnostic_layer_attention(image_attentions)
        text_attention, text_attention_source = self._diagnostic_layer_attention(text_attentions)

        image_feats = image_tokens[:, 0, :].float()
        rows = torch.arange(text_tokens.shape[0], device=text_tokens.device)
        text_feats = text_tokens[rows, caption_ids.argmax(dim=-1)].float()
        features = {"i_feats": image_feats, "t_feats": text_feats}
        use_local = bool(branch.use_local and not self.args.only_global)
        if use_local:
            features["i_grab_f"] = self.visul_emb_layer(image_tokens, image_attention.clone(), current_step).float()
            features["t_grab_f"] = self.texual_emb_layer(text_tokens, caption_ids, text_attention.clone(), current_step).float()

        proto_image_feats, proto_text_feats = self._select_prototype_features(features)
        use_local = bool(branch.use_local and not self.args.only_global)
        visual_prior, visual_mask = self._visual_evidence_prior(image_attention, use_local, current_step=current_step)
        text_prior, text_mask = self._text_evidence_prior(text_attention, caption_ids, use_local, current_step=current_step)
        evidence = branch.evidence(
            proto_image_feats,
            proto_text_feats,
            pids,
            image_token_features=self._visual_tokens_for_prototype_evidence(image_tokens),
            text_token_features=self._text_tokens_for_prototype_evidence(text_tokens),
            image_attention_prior=visual_prior,
            text_attention_prior=text_prior,
            image_token_mask=visual_mask,
            text_token_mask=text_mask,
            max_prototypes_per_id=max_prototypes_per_id,
        )
        visual = getattr(self.base_model, "visual", None)
        evidence.update({
            "visual_token_mask": visual_mask.detach(),
            "text_token_mask": text_mask.detach(),
            "image_attention_prior": visual_prior.detach(),
            "text_attention_prior": text_prior.detach(),
            "image_grid": (
                int(getattr(visual, "num_y", 0) or 0),
                int(getattr(visual, "num_x", 0) or 0),
            ),
            "prototype_feature_source": "local" if use_local else "global",
            "image_attention_source": image_attention_source,
            "text_attention_source": text_attention_source,
        })
        if include_raw_heads:
            evidence["raw_image_attentions"] = [a.detach() for a in image_attentions]
            evidence["raw_text_attentions"] = [a.detach() for a in text_attentions]
        return evidence


    @torch.no_grad()
    def extract_prototype_features(self, batch, current_step=None):
        features = self._compute_host_embeddings(batch['images'], batch['caption_ids'], current_step=current_step)
        return self._select_prototype_features(features)

    def forward(self, batch, epoch=None, current_step=None):
        ret = dict()
        device = "cuda"

        if 'cid' in self.current_task:
            self.mlp_global = self.mlp_global.float()
            self.classifier_global = self.classifier_global.float()
            if not self.args.only_global:
                self.mlp_grab = self.mlp_grab.float()
                self.classifier_grab = self.classifier_grab.float()
        
        ret.update({'temperature': 1 / self.logit_scale})
        images = batch['images']
        caption_ids = batch['caption_ids']
        features = self._compute_host_embeddings(images, caption_ids, current_step=current_step)
        i_feats = features["i_feats"]
        t_feats = features["t_feats"]
        if not self.args.only_global:
            i_grab_f = features["i_grab_f"]
            t_grab_f = features["t_grab_f"]

        if getattr(self.args, "track_train_diagnostics", True):
            if not self.args.only_global:
                host_image_feats, host_text_feats = i_grab_f, t_grab_f
            else:
                host_image_feats, host_text_feats = i_feats, t_feats
            proto_image_feats, proto_text_feats = self._select_prototype_features(features)
            ret["_diag"] = {
                "host_image_feats": host_image_feats.detach(),
                "host_text_feats": host_text_feats.detach(),
                "proto_image_feats": proto_image_feats.detach(),
                "proto_text_feats": proto_text_feats.detach(),
                "pids": batch["pids"].detach(),
                "indices": batch.get("index", None),
            }

        if 'cid' in self.current_task:
            S = objectives.cosine_similarity_matrix(i_feats, t_feats)
            hard_negatives = objectives.sample_hard_negatives(S, batch['pids'])
            M = batch['pids'].max().item()
            new_labels = objectives.update_labels_for_negatives(batch['pids'], hard_negatives, M)
            all_pairs = objectives.create_sample_pairs(i_feats, t_feats, hard_negatives, new_labels, batch['pids'])
            ni_feats, nt_feats, nlabels = all_pairs
            z_feats1 = torch.cat([ni_feats.float(), nt_feats.float()], dim=1)
            z_feats2 = torch.cat([nt_feats.float(), ni_feats.float()], dim=1)
            z_feats1 = self.mlp_global(z_feats1.float())
            z_feats2 = self.mlp_global(z_feats2.float())
            cross_modal_logits1 = self.classifier_global(z_feats1.float())
            cross_modal_logits2 = self.classifier_global(z_feats2.float())
            device = cross_modal_logits1.device 
            nlabels = nlabels.to(device) 
            closs1 =  objectives.compute_cid(cross_modal_logits1, cross_modal_logits2,nlabels)
            image_logits = self.classifier_id_global(i_feats.half()).float()
            text_logits = self.classifier_id_global(t_feats.half()).float()
            closs3 = objectives.compute_id(image_logits, batch['pids']) + objectives.compute_id(text_logits, batch['pids'])
            
            if not self.args.only_global:
                S_ = objectives.cosine_similarity_matrix(i_grab_f, t_grab_f)
                hard_negatives_ = objectives.sample_hard_negatives(S_, batch['pids'])
                M_ = batch['pids'].max().item()
                new_labels_ = objectives.update_labels_for_negatives(batch['pids'], hard_negatives_, M_)
                all_pairs_ = objectives.create_sample_pairs(i_grab_f, t_grab_f, hard_negatives_, new_labels_, batch['pids'])
                ni_feats_, nt_feats_, nlabels_ = all_pairs_
                z_feats1_ = torch.cat([ni_feats_.float(), nt_feats_.float()], dim=1)
                z_feats2_ = torch.cat([nt_feats_.float(), ni_feats_.float()], dim=1)
                z_feats1_ = self.mlp_grab(z_feats1_.float())
                z_feats2_ = self.mlp_grab(z_feats2_.float())
                cross_modal_logits1_ = self.classifier_grab(z_feats1_.float())
                cross_modal_logits2_ = self.classifier_grab(z_feats2_.float())
                nlabels_ = nlabels_.to(device)
                closs2 =  objectives.compute_cid(cross_modal_logits1_, cross_modal_logits2_,nlabels_)
                image_logits_ = self.classifier_id_grab(i_grab_f.half()).float()
                text_logits_ = self.classifier_id_grab(t_grab_f.half()).float()
                closs4 = objectives.compute_id(image_logits_, batch['pids']) + objectives.compute_id(text_logits_, batch['pids'])
                ret.update({'cid_loss': closs1+closs2+closs3+closs4})
            else:
                ret.update({'cid_loss': closs1+closs3})

        if 'tal' in self.current_task:
            TAL_global_loss = objectives.compute_TAL(i_feats, t_feats,batch['pids'],margin=self.args.margin,tau=self.args.tau)
            if not self.args.only_global:
                TAL_grab_loss = objectives.compute_TAL(i_grab_f, t_grab_f,batch['pids'],margin=self.args.margin,tau=self.args.tau)
                ret.update({'tal_loss': TAL_global_loss + TAL_grab_loss}) 
            else:
                ret.update({'tal_loss': TAL_global_loss})

        if self.prototype_enabled and self.prototype_branch is not None:
            proto_image_feats, proto_text_feats = self._select_prototype_features(features)
            proto_ret = self.prototype_branch(
                proto_image_feats,
                proto_text_feats,
                batch['pids'],
                use_loss_id=getattr(self.args, "use_loss_id", False),
            )
            if "proto_id_loss" in proto_ret:
                ret["proto_id_loss"] = proto_ret["proto_id_loss"] * getattr(self.args, "prototype_id_weight", 0.2)

        return ret

def build_model(args, num_classes=11003):
    model = ITSELF(args, num_classes)
    convert_weights(model)

    return model
