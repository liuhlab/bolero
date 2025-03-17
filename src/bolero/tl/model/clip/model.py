from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from torchmetrics import AUROC


@contextmanager
def _null_context():
    yield


def _model_forward_with_context(
    *,
    fn,
    args,
    freeze,
):
    encoding_context = _null_context if not freeze else torch.no_grad

    with encoding_context(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        enc = fn(*args)
        if hasattr(enc, "embeddings"):
            enc = enc.embeddings

        if freeze:
            enc = enc.detach()

    return enc


def _matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device=device)
    j_range = torch.arange(j, device=device)
    diag_mask = rearrange(i_range, "i -> i 1") == rearrange(j_range, "j -> 1 j")
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, "(b d) -> b d", d=num_diag_el)


def log(t, eps=1e-20):
    """Log with eps"""
    return torch.log(t + eps)


class CLIP(nn.Module):
    def __init__(
        self,
        encoder,
        freeze_encoder=True,
        emb_dim=1152,
        latent_dim=1152,
        temperature_init=0.5,
    ):
        super().__init__()
        self.encoder = encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            encoder.eval()
        self.freeze_encoder = freeze_encoder

        self.temperature = nn.Parameter(
            torch.tensor(temperature_init, dtype=torch.float32)
        )

        self.text_projection = nn.Linear(emb_dim, latent_dim)
        self.image_projection = nn.Linear(emb_dim, latent_dim)

    @staticmethod
    def _calc_auc(text_to_image, batch_size):
        with torch.no_grad():
            flat_similarity_matrix = text_to_image.clone().view(-1)
            auc_labels = torch.zeros(batch_size * batch_size).to(text_to_image)
            auc_labels[torch.arange(batch_size) * (batch_size + 1)] = 1
            auroc = AUROC(task="binary")
            contra_auc = torch.tensor(
                [auroc(flat_similarity_matrix, auc_labels).item()]
            ).to(text_to_image)
        return contra_auc

    @staticmethod
    def _accuracy(output, batch_size, topk):
        target = torch.tensor(list(range(batch_size))).to(output)
        with torch.no_grad():
            maxk = max(topk)
            size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)

            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(1.0 / size))
        return res

    @staticmethod
    def _calculate_cl_loss(text_to_image, image_to_text):
        # exponentiate
        text_to_image_exp, image_to_text_exp = map(
            torch.exp, (text_to_image, image_to_text)
        )
        # numerators
        text_to_image_pos, image_to_text_pos = map(
            _matrix_diag, (text_to_image_exp, image_to_text_exp)
        )

        # denominator
        text_to_image_denom, image_to_text_denom = (
            t.sum(dim=-1) for t in (text_to_image_exp, image_to_text_exp)
        )

        # calculate CL loss
        text_to_image_loss = (-log(text_to_image_pos) + log(text_to_image_denom)).mean(
            dim=-1
        )
        image_to_text_loss = (-log(image_to_text_pos) + log(image_to_text_denom)).mean(
            dim=-1
        )
        cl_losses = (text_to_image_loss + image_to_text_loss) / 2
        return cl_losses

    def forward(self, x):
        """Forward pass"""
        emb = _model_forward_with_context(
            fn=self.encoder,
            args=(x,),
            freeze=self.freeze_encoder,
        )

        # Split and take CLS Token
        batch_size = emb.shape[0] // 2
        emb1 = emb[:batch_size, 0, :]
        emb2 = emb[batch_size:, 0, :]

        # Similarity Matrix
        text_emb = self.text_projection(emb1)  # (batch_size, latent_dim)
        text_emb = F.normalize(text_emb, dim=-1)
        img_emb = self.image_projection(emb2)  # (batch_size, latent_dim)
        img_emb = F.normalize(img_emb, dim=-1)
        temp = self.temperature.exp()
        text_to_image = einsum(text_emb, img_emb, "t d, i d -> t i") * temp
        image_to_text = rearrange(text_to_image, "t i -> i t")

        # calculate loss
        # exponentiate
        text_to_image_exp, image_to_text_exp = map(
            torch.exp, (text_to_image, image_to_text)
        )
        # numerators
        text_to_image_pos, image_to_text_pos = map(
            _matrix_diag, (text_to_image_exp, image_to_text_exp)
        )
        # denominator
        text_to_image_denom, image_to_text_denom = (
            t.sum(dim=-1) for t in (text_to_image_exp, image_to_text_exp)
        )
        # calculate CL loss
        text_to_image_loss = (-log(text_to_image_pos) + log(text_to_image_denom)).mean(
            dim=-1
        )
        image_to_text_loss = (-log(image_to_text_pos) + log(image_to_text_denom)).mean(
            dim=-1
        )
        cl_losses = (text_to_image_loss + image_to_text_loss) / 2

        # AUC
        with torch.no_grad():
            flat_similarity_matrix = text_to_image.clone().view(-1)
            auc_labels = torch.zeros(batch_size * batch_size).to(text_to_image)
            auc_labels[torch.arange(batch_size) * (batch_size + 1)] = 1
            auroc = AUROC(task="binary")
            contra_auc = torch.tensor(
                [auroc(flat_similarity_matrix, auc_labels).item()]
            ).to(text_to_image)

        # ACC
        def accuracy(output, target, topk):
            with torch.no_grad():
                maxk = max(topk)
                size = target.size(0)
                _, pred = output.topk(maxk, 1, True, True)
                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))
                res = []
                for k in topk:
                    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                    res.append(correct_k.mul_(1.0 / size))
                return res

        acc_labels = torch.tensor(list(range(batch_size))).to(text_to_image)
        contra_top1_acc, contra_top5_acc = accuracy(
            text_to_image, acc_labels, topk=(1, 5)
        )

        result = {
            "loss": cl_losses,
            "auc": contra_auc,
            "top1_acc": contra_top1_acc,
            "top5_acc": contra_top5_acc,
        }
        return result

    def _get_text_logits(self, text_emb, img_emb):
        """Get logits from text and image embeddings."""
        temp = self.temperature.exp()
        text_emb = self.text_projection(text_emb)
        text_emb = F.normalize(text_emb, dim=-1)
        img_emb = self.image_projection(img_emb)
        img_emb = F.normalize(img_emb, dim=-1)
        text_logits = einsum(text_emb, img_emb, "t d, i d -> t i") * temp
        return text_logits

    def _get_img_logits(self, img_emb, text_emb):
        """Get logits from image and text embeddings."""
        temp = self.temperature.exp()
        img_emb = self.image_projection(img_emb)
        img_emb = F.normalize(img_emb, dim=-1)
        text_emb = self.text_projection(text_emb)
        text_emb = F.normalize(text_emb, dim=-1)
        img_logits = einsum(img_emb, text_emb, "i d, t d -> i t") * temp
        return img_logits

    @torch.no_grad()
    def _get_logits(self, query_emb, target_emb):
        """Get logits from query and target embeddings."""
        text_logits = self._get_text_logits(query_emb, target_emb)
        # img_logits = self._get_img_logits(query_emb, target_emb)
        # logits = (text_logits + img_logits) / 2
        return text_logits

    @torch.no_grad()
    def embedding(self, x):
        """Get embeddings from a trained CLIP model."""
        emb = _model_forward_with_context(
            fn=self.encoder,
            args=(x,),
            freeze=self.freeze_encoder,
        )
        return emb
