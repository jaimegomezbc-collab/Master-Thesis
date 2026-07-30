import os
from pathlib import Path

os.environ["TORCH_EXTENSIONS_DIR"] = str(Path(os.environ["CONDA_PREFIX"]) / "torch_extensions")
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
from backbones.ncsnpp_generator_adagn_feat import NCSNpp
from backbones.ncsnpp_generator_adagn_feat import NCSNpp_adaptive
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse
import torch
import torchvision.transforms
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as cm
from skimage.segmentation import mark_boundaries
from torch.utils.checkpoint import checkpoint
from captum.attr import Saliency, IntegratedGradients, NoiseTunnel
import time
import json



def modality_score(sal):
    # sum over spatial dims, optionally also channels
    return sal.sum(dim=(1, 2, 3), keepdim=True)  # or (2,3) if already [1,1,H,W]


def load_checkpoint(checkpoint_dir, netG, name_of_network, device='cuda:0'):
    checkpoint_file = checkpoint_dir.format(name_of_network)

    checkpoint = torch.load(checkpoint_file, map_location=device)
    ckpt = checkpoint

    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    netG.load_state_dict(ckpt, strict=False)
    netG.eval()


# %% Diffusion coefficients
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out


def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    return t.to(device)


def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small

    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas ** 0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas


# %% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)

        # we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32, device=device), self.alphas_cumprod[:-1]), 0
        )
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
                (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))

        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))


def sample_posterior(coefficients, x_0, x_t, t):
    def q_posterior(x_0, x_t, t):
        mean = (
                extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
                + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)

        noise = torch.randn_like(x_t)

        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0, x_t, t)

    return sample_x_pos


def sample_posterior_combine(coefficients, x_0_1, x_0_2, x_t, t):
    def q_posterior(x_0_1, x_0_2, x_t, t):
        mean1 = (
                extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0_1
                + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        mean2 = (
                extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0_2
                + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        mean = (mean1 + mean2) / 2

        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0_1, x_0_2, x_t, t):
        mean, _, log_var = q_posterior(x_0_1, x_0_2, x_t, t)

        noise = torch.randn_like(x_t)

        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0_1, x_0_2, x_t, t)

    return sample_x_pos


class DiffusionAttributor:
    def __init__(self, generator1, generator2):
        self.generator1 = generator1
        self.generator2 = generator2

    def forward_outputs(
            self,
            cond1_in,
            cond2_in,
            cond3_in,
            *,
            x,
            t_time,
            latent_z,
    ):
        B_ig = cond1_in.shape[0]

        # Broadcast x and latent_z along batch dimension
        x_step = x.detach().expand(B_ig, -1, -1, -1)  # [B_ig, 1, H, W]
        latent_z_step = latent_z.detach().expand(B_ig, -1)  # [B_ig, nz]

        x_0_1 = checkpoint(
            lambda a, b, c, d, e, f: self.generator1(a, b, c, d, e, f),
            x_step,
            cond1_in,
            cond2_in,
            cond3_in,
            t_time,
            latent_z_step,
            use_reentrant=False,
        )

        x_0_2 = checkpoint(
            lambda a, b, c, d, e, f, g: self.generator2(a, b, c, d, e, f, g),
            x_step,
            cond1_in,
            cond2_in,
            cond3_in,
            t_time,
            latent_z_step,
            x_0_1[:, [0], :],
            use_reentrant=False,
        )

        return x_0_1, x_0_2

    def forward_score(
            self,
            cond1_in,
            cond2_in,
            cond3_in,
            *,
            x,
            t_time,
            latent_z,
            roi_mask=None,
    ):
        _, x_0_2 = self.forward_outputs(
            cond1_in,
            cond2_in,
            cond3_in,
            x=x,
            t_time=t_time,
            latent_z=latent_z,
        )

        target = x_0_2[:, [0], :]

        if roi_mask is None:
            score = target.sum()
        else:
            roi_mask = roi_mask.to(target.device)
            score = (target * roi_mask).sum()

            # Make it 1-D of length 1 so Captum's gradient utils can index outputs[0]
        return score.unsqueeze(0)

    def make_forward_func(self, *, x, t_time, latent_z, roi_mask=None):
        def forward_func(cond1_in, cond2_in, cond3_in):
            return self.forward_score(
                cond1_in,
                cond2_in,
                cond3_in,
                x=x,
                t_time=t_time,
                latent_z=latent_z,
                roi_mask=roi_mask,
            )

        return forward_func


def _reduce_attr(attr, positive_only=False):
    with torch.no_grad():
        if positive_only:
            attr = torch.clamp(attr, min=0.0)
            return attr.mean().item()
        return attr.abs().mean().item()

def _normalize_map(sm):
    sm = sm.detach()
    #thresh = torch.quantile(sm.flatten(), 0.95)
    #sm = torch.where(sm < thresh, torch.zeros_like(sm), sm)
    sm_min = sm.min()
    sm_max = sm.max()
    sm_out = sm / (sm_max - sm_min + 1e-8)
    return sm_out



def sample_from_model(
        coefficients,
        generator1,
        cond1,
        generator2,
        cond2,
        cond3,
        n_time,
        x_init,
        T,
        opt,
        track_modality_contrib=False,
        step_stride=None,
        roi_mask=None,
        positive_only=False,
        return_saliency_maps=False,
        normalize_saliency_maps=True,
        attribution_method="saliency",  # "saliency" or "ig"
        ig_steps=64,
        sg_nt_samples=8,
        sg_nt_samples_batch_size=2,
        sg_stdevs=0.10,
    ):
    x = x_init
    attributor = DiffusionAttributor(generator1, generator2)
    methods = tuple(attribution_method)
    valid_methods = {"saliency", "IntGrad", "SmoothGrad"}
    for m in methods:
        if m not in valid_methods:
            raise ValueError(f"Unknown attribution method: {m}")

    with torch.no_grad():
        unc_accum = torch.zeros_like(x_init)

    modality_scores = {
        m:{"flair": 0.0,"t2": 0.0, "t1": 0.0}
        for m in methods
    }

    saliency_maps = None
    saliency_steps = {m: 0 for m in methods}

    if return_saliency_maps:
        saliency_maps = {
            m: {
                "flair": torch.zeros_like(cond1),
                "t2": torch.zeros_like(cond2),
                "t1": torch.zeros_like(cond3),
            }
            for m in methods
        }

    if step_stride is None:
        step_stride = 1 if track_modality_contrib else None

    for i in reversed(range(n_time)):
        print(i)
        t = torch.full((x.size(0),), i, dtype=torch.int64, device=x.device)
        t_time = t
        latent_z = torch.randn(x.size(0), opt.nz, device=x.device)

        do_attr = (
                track_modality_contrib
                and (step_stride is not None)
                and (i % step_stride == 0)
        )

        if do_attr:
            cond1_step = cond1.detach().clone().requires_grad_(True)
            cond2_step = cond2.detach().clone().requires_grad_(True)
            cond3_step = cond3.detach().clone().requires_grad_(True)

            inputs = (cond1_step, cond2_step, cond3_step)
            baselines = tuple(torch.zeros_like(inp) for inp in inputs)

            forward_func = attributor.make_forward_func(
                x=x,
                t_time=t_time,
                latent_z=latent_z,
                roi_mask=roi_mask,
            )

            attrs = {}

            if "saliency" or "SmoothGrad" in methods:
                sal = Saliency(forward_func)

                if "saliency" in methods:
                    attrs["saliency"] = sal.attribute(
                        inputs=inputs,
                        abs=False,
                    )

                if "SmoothGrad" in methods:
                    nt_sal = NoiseTunnel(sal)
                    attrs["SmoothGrad"] = nt_sal.attribute(
                        inputs=inputs,
                        nt_type="smoothgrad",
                        nt_samples=sg_nt_samples,
                        nt_samples_batch_size=sg_nt_samples_batch_size,
                        stdevs=sg_stdevs,
                        abs=False,
                    )

            if "IntGrad" in methods:
                ig = IntegratedGradients(forward_func)
                attrs["IntGrad"] = ig.attribute(
                    inputs=inputs,
                    baselines=baselines,
                    n_steps=ig_steps,
                    method="gausslegendre",
                )

            for method_name, (a1, a2, a3) in attrs.items():
                modality_scores[method_name]["flair"] += _reduce_attr(a1, positive_only=positive_only)
                modality_scores[method_name]["t2"] += _reduce_attr(a2, positive_only=positive_only)
                modality_scores[method_name]["t1"] += _reduce_attr(a3, positive_only=positive_only)

                if return_saliency_maps:
                    saliency_maps[method_name]["flair"] += a1.detach()
                    saliency_maps[method_name]["t2"] += a2.detach()
                    saliency_maps[method_name]["t1"] += a3.detach()
                    saliency_steps[method_name] += 1

            with torch.no_grad():
                x_0_1, x_0_2 = attributor.forward_outputs(
                    cond1,
                    cond2,
                    cond3,
                    x=x,
                    t_time=t_time,
                    latent_z=latent_z,
                )

                unc_map_t = torch.abs(x_0_1[:, [0], :] - x_0_2[:, [0], :])
                unc_accum = unc_accum + unc_map_t

                x = sample_posterior_combine(
                    coefficients,
                    x_0_1[:, [0], :].detach(),
                    x_0_2[:, [0], :].detach(),
                    x,
                    t,
                ).detach()

                del x_0_1, x_0_2, unc_map_t
            del cond1_step, cond2_step, cond3_step, attrs, baselines, inputs

        else:
            with torch.no_grad():
                x_0_1 = generator1(x, cond1, cond2, cond3, t_time, latent_z)
                x_0_2 = generator2(x, cond1, cond2, cond3, t_time, latent_z, x_0_1[:, [0], :])

                unc_map_t = torch.abs(x_0_1[:, [0], :] - x_0_2[:, [0], :])
                unc_accum = unc_accum + unc_map_t

                x = sample_posterior_combine(
                    coefficients,
                    x_0_1[:, [0], :].detach(),
                    x_0_2[:, [0], :].detach(),
                    x,
                    t,
                ).detach()

                del x_0_1, x_0_2, unc_map_t

        del latent_z, t, t_time

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    unc = unc_accum / n_time

    if track_modality_contrib:
        modality_scores_norm = {}
        for m in methods:
            total = (
                    modality_scores[m]["flair"]
                    + modality_scores[m]["t2"]
                    + modality_scores[m]["t1"]
                    + 1e-8
            )
            modality_scores_norm[m] = {
                "flair": modality_scores[m]["flair"] / total,
                "t2": modality_scores[m]["t2"] / total,
                "t1": modality_scores[m]["t1"] / total,
            }

        if return_saliency_maps:
            for m in methods:
                if saliency_steps[m] > 0:
                    saliency_maps[m]["flair"] /= saliency_steps[m]
                    saliency_maps[m]["t2"] /= saliency_steps[m]
                    saliency_maps[m]["t1"] /= saliency_steps[m]

                if normalize_saliency_maps:
                    for k in saliency_maps[m]:
                        saliency_maps[m][k] = _normalize_map(saliency_maps[m][k])

            return x, unc, modality_scores, modality_scores_norm, saliency_maps

        return x, unc, modality_scores, modality_scores_norm

    return x, unc


# Normalize the image using min-max scaling
def normalize(image):
    """Basic min max scaler."""
    min_ = np.min(image)
    max_ = np.max(image)
    scale = max_ - min_
    image = (image - min_) / scale
    return image


# Pre-process using IRM min-max scaling
def irm_min_max_preprocess(image, low_perc=1, high_perc=99):
    """Main pre-processing function for removing outliers and scaling."""
    non_zeros = image > 0
    low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
    image = np.clip(image, low, high)
    image = normalize(image)
    return image


# Load and preprocess a single image
def tensorize_image(img, size=(256, 256)):
    """
    Converts np into tensor and normalizes
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size),
        torchvision.transforms.ToTensor()  # This will convert the image to a tensor of shape (1, 256, 256)
    ])

    # Apply IRM min-max pre-processing
    img = irm_min_max_preprocess(img)

    # Normalize to [-1, 1] by applying (data - 0.5) / 0.5
    img = (img - 0.5) / 0.5

    # Convert back to tensor and add batch and channel dimensions
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 256, 256)

    return img_tensor

def to_jsonable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    return obj


if __name__ == "__main__":
    device = 'cuda:0'
    # Define all arguments as a Namespace for use in Jupyter Notebook
    args = argparse.Namespace(
        # General setup
        seed=1024,
        compute_fid=False,
        epoch_id=1000,

        # Diffusion process parameters
        num_timesteps=4,
        beta_min=0.1,
        beta_max=20.0,
        centered=True,
        use_geometric=False,

        # Model architecture parameters
        num_channels=1,
        num_channels_dae=64,
        n_mlp=3,
        ch_mult=[1, 2, 4],
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        conditional=True,
        fir=True,
        fir_kernel=[1, 3, 3, 1],
        skip_rescale=True,
        resblock_type='biggan',
        progressive='none',
        progressive_input='residual',
        progressive_combine='sum',
        embedding_type='positional',
        fourier_scale=16.0,
        not_use_tanh=False,

        # Experiment setup
        exp='ixi_synth',
        input_path='./input',
        output_path='./output',
        dataset='cifar10',
        image_size=256,

        # Generator-specific parameters
        nz=100,
        z_emb_dim=256,
        t_emb_dim=256,
        batch_size=1,

        # Optimizer parameters
        lr_g=1.5e-4,
        beta1=0.5,
        beta2=0.9,

        # Hardware configuration
        gpu_chose=0,
    )

    # Example of modifying an argument interactively
    args.num_timesteps = 1000
    args.exp = 'experiment_name'
    print('Prep 1: Done')

    gen_diffusive_1 = NCSNpp(args).to(device)
    gen_diffusive_2 = NCSNpp_adaptive(args).to(device)
    gen_diffusive_1.eval()
    gen_diffusive_2.eval()

    # Load checkpoints
    load_checkpoint(r'MU-Diff_Model_Weights/brats/t1ce/gen_diffusive_1.pth', gen_diffusive_1, 'gen_diffusive_1',
                    device=device)
    load_checkpoint(r'MU-Diff_Model_Weights/brats/t1ce/gen_diffusive_2.pth', gen_diffusive_2, 'gen_diffusive_2',
                    device=device)

    print('Prep 2: Done')

    image_folder = "/cs/student/project_msc/2025/aibh/jgomezbe/UCSD-PTGBM/middle_slices"
    contrast_folder = os.path.join(image_folder, "cet1")

    for image_name in os.listdir(contrast_folder):
        curr_fold = Path(image_folder)
        output_dir = os.path.join("/cs/student/project_msc/2025/aibh/jgomezbe/Master-Thesis/Explanations",image_name[:-4])
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(os.path.join(output_dir, "modality_captum_SmoothGrad.png")):
            print(f"Image {image_name[:-4]} already explained")
            continue
        x1_path = os.path.join(image_folder,os.path.join("flair",image_name))
        x2_path = os.path.join(image_folder,os.path.join("t2",image_name))
        x3_path = os.path.join(image_folder,os.path.join("t1",image_name))
        real_data_path = os.path.join(image_folder,os.path.join("cet1",image_name))
        enhancement_mask_path = os.path.join(image_folder, os.path.join("enhanced_regions_mask", image_name))
        # Load images
        x1 = tensorize_image(np.load(x1_path)).cuda()
        x2 = tensorize_image(np.load(x2_path)).cuda()
        x3 = tensorize_image(np.load(x3_path)).cuda()
        real_data = tensorize_image(np.load(real_data_path)).cuda()
        enhancement_mask = np.load(enhancement_mask_path)

        roi_mask = torch.tensor(enhancement_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 256, 256)
        sample_inputs = torch.cat((x1, x2, x3, real_data), axis=-1)  # Concatenate along the width

        sample_inputs = sample_inputs.squeeze(0).squeeze(0)  # Shape: (256, 256, 5)

        T = get_time_schedule(args, device)
        pos_coeff = Posterior_Coefficients(args, device)

        # Initialize noisy input
        x1_t = torch.randn_like(real_data)
        print(f'Prep 3: Done. Current image: {image_name[:-4]}')
        fake_sample, unc, modality_scores_raw, modality_scores_norm, saliency_maps = sample_from_model(
            pos_coeff,
            gen_diffusive_1,
            x1,
            gen_diffusive_2,
            x2,
            x3,
            args.num_timesteps,
            x1_t,
            T,
            args,
            track_modality_contrib=True,
            step_stride=1,  # 10 sampled timesteps over 1000
            roi_mask=roi_mask,  # or lesion mask
            positive_only=False,  # abs gradients
            return_saliency_maps=True,
            normalize_saliency_maps=True,
            attribution_method=["saliency","SmoothGrad","IntGrad"],  # "saliency" or "ig"
            ig_steps=4,
            sg_nt_samples=8,
            sg_nt_samples_batch_size=2,
            sg_stdevs=0.10,
        )

        for method in modality_scores_raw:
            print(f"\nRaw modality scores {method}")
            print("FLAIR:", round(modality_scores_raw[method]["flair"], 4))
            print("T2:   ", round(modality_scores_raw[method]["t2"], 4))
            print("T1:   ", round(modality_scores_raw[method]["t1"], 4))
            print(f"\nNormalized modality fractions {method}")
            print("FLAIR:", round(modality_scores_norm[method]["flair"], 4))
            print("T2:   ", round(modality_scores_norm[method]["t2"], 4))
            print("T1:   ", round(modality_scores_norm[method]["t1"], 4))
        with open(os.path.join(output_dir, "modality_scores_raw.json"), "w") as f:
            json.dump(to_jsonable(modality_scores_raw), f, indent=2)
        with open(os.path.join(output_dir, "modality_scores_norm.json"), "w") as f:
            json.dump(to_jsonable(modality_scores_norm), f, indent=2)

        unc = unc - unc.min()
        unc_max = unc.max()
        unc = unc / (unc_max + 1e-8)

        # Normalize and save
        to_range_0_1 = lambda x: (x + 1.) / 2.
        fake_sample = to_range_0_1(fake_sample)

        fake_sample = fake_sample * 255.0
        fake_sample = fake_sample.squeeze(0).squeeze(0)  # Shape: (256, 256, 5)

        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.imshow(real_data.squeeze(0).squeeze(0).cpu().numpy(), cmap='gray')  # Display in grayscale
        plt.axis('off')  # Hide axes
        plt.set_title(f"Real CE-T1")
        plt.subplot(122)
        plt.imshow(fake_sample.cpu().numpy(), cmap='gray')  # Display in grayscale
        plt.axis('off')  # Hide axes
        plt.set_title(f"Synthetized CE-T1")
        plt.savefig(os.path.join(output_dir,"Generated Image.png"), bbox_inches="tight", pad_inches=0)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(fake_sample.cpu().numpy(), cmap='gray')
        ax.imshow(unc.squeeze(0).squeeze(0).cpu().numpy(), cmap='jet', alpha=0.35)
        ax.axis('off')

        cbar = plt.colorbar(ax.images[1], ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Uncertainty')

        proxy = Line2D([], [], color='none', label=f'Maximum uncertainty: {unc_max:.4f}')
        ax.legend(handles=[proxy], loc='lower left', frameon=True)

        plt.savefig(os.path.join(output_dir, "uncertainty_overlay.png"),
                    bbox_inches='tight', pad_inches=0.1)
        overlap_score = {}

        for method in modality_scores_raw:
            curr_overlap_score = 0
            normalization_const = 0
            fig = plt.figure(figsize=(12, 8), constrained_layout=True)
            gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.06])

            ax00 = fig.add_subplot(gs[0, 0])
            ax01 = fig.add_subplot(gs[0, 1])
            ax10 = fig.add_subplot(gs[1, 0])
            ax11 = fig.add_subplot(gs[1, 1])
            cax = fig.add_subplot(gs[:, 2])

            all_sal = np.concatenate([
                saliency_maps[method]["flair"].squeeze(0).squeeze(0).cpu().numpy().ravel(),
                saliency_maps[method]["t2"].squeeze(0).squeeze(0).cpu().numpy().ravel(),
                saliency_maps[method]["t1"].squeeze(0).squeeze(0).cpu().numpy().ravel(),
            ])

            nonzero_abs = np.abs(all_sal[np.abs(all_sal) > 0])
            vmax = np.percentile(nonzero_abs, 99.7) if len(nonzero_abs) > 0 else np.max(np.abs(all_sal))
            if vmax == 0:
                vmax = 1e-8

            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

            thr = 0.2 * vmax

            cmap = cm.get_cmap('seismic').copy()
            cmap.set_bad((0, 0, 0, 0))

            flair_sal = saliency_maps[method]["flair"].squeeze(0).squeeze(0).cpu().numpy().copy()
            t2_sal = saliency_maps[method]["t2"].squeeze(0).squeeze(0).cpu().numpy().copy()
            t1_sal = saliency_maps[method]["t1"].squeeze(0).squeeze(0).cpu().numpy().copy()

            flair_sal[np.abs(flair_sal) < thr] = np.nan
            t2_sal[np.abs(t2_sal) < thr] = np.nan
            t1_sal[np.abs(t1_sal) < thr] = np.nan

            outlined = mark_boundaries(real_data.squeeze(0).squeeze(0).cpu().numpy(), enhancement_mask, color=(1, 1, 0), mode='thick')
            ax00.imshow(real_data.squeeze(0).squeeze(0).cpu().numpy(), cmap="grey")
            ax00.imshow(outlined, alpha= 0.35)
            ax00.set_title("Real CE-T1")

            outlined = mark_boundaries(x1.squeeze(0).squeeze(0).cpu().numpy(), enhancement_mask, color=(1, 1, 0),
                                       mode='thick')
            ax01.imshow(x1.squeeze(0).squeeze(0).cpu().numpy(), cmap="grey")
            ax01.imshow(outlined, alpha= 0.35)
            im1 = ax01.imshow(flair_sal, cmap=cmap, alpha=0.45, norm=norm)
            ax01.set_title(f"FLAIR {method}")

            curr_overlap_score += np.sum(
                saliency_maps[method]["flair"].squeeze(0).squeeze(0).cpu().numpy() * enhancement_mask)
            normalization_const += np.sum(enhancement_mask)

            outlined = mark_boundaries(x2.squeeze(0).squeeze(0).cpu().numpy(), enhancement_mask, color=(1, 1, 0),
                                       mode='thick')
            ax10.imshow(x2.squeeze(0).squeeze(0).cpu().numpy(), cmap="grey")
            ax10.imshow(outlined, alpha= 0.35)
            ax10.imshow(t2_sal, cmap=cmap, alpha=0.45, norm=norm)
            ax10.set_title(f"T2 {method}")
            curr_overlap_score += np.sum(
                saliency_maps[method]["t2"].squeeze(0).squeeze(0).cpu().numpy() * enhancement_mask)
            normalization_const += np.sum(enhancement_mask)

            outlined = mark_boundaries(x3.squeeze(0).squeeze(0).cpu().numpy(), enhancement_mask, color=(1, 1, 0),
                                       mode='thick')
            ax11.imshow(x3.squeeze(0).squeeze(0).cpu().numpy(), cmap="grey")
            ax11.imshow(outlined, alpha= 0.35)
            ax11.imshow(t1_sal, cmap=cmap, alpha=0.45, norm=norm)
            ax11.set_title(f"T1 {method}")
            curr_overlap_score += np.sum(
                saliency_maps[method]["t1"].squeeze(0).squeeze(0).cpu().numpy() * enhancement_mask)
            normalization_const += np.sum(enhancement_mask)

            for ax in [ax00, ax01, ax10, ax11]:
                ax.axis("off")

            fig.colorbar(im1, cax=cax)
            plt.savefig(os.path.join(output_dir,f"modality_captum_{method}.png"), dpi=200, bbox_inches="tight")
            # I NEED TO NORMALIZE THIS SCORE (MAP VALUES FROM -1 TO 1, BUT SUM MIGHT BE HIGHER THAN 1)
            overlap_score[method] = curr_overlap_score/normalization_const
            print(f"Overlap score of {method} for {image_name[:-4]}: {curr_overlap_score}")
        with open(os.path.join(output_dir,"overlap_metric.json"), "w") as f:
            json.dump(to_jsonable(overlap_score), f, indent=2)