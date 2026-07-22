import os
from pathlib import Path
os.environ["TORCH_EXTENSIONS_DIR"] = str(Path(os.environ["CONDA_PREFIX"]) / "torch_extensions")
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
from backbones.ncsnpp_generator_adagn_feat import NCSNpp
from backbones.ncsnpp_generator_adagn_feat import NCSNpp_adaptive
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from torch.utils.checkpoint import checkpoint
from captum.attr import Saliency, IntegratedGradients

def select_roi_interactive(image_2d):
    coords = {}

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_2d, cmap='gray')
    ax.set_title("Drag ROI, then close window")
    ax.axis('off')

    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        coords['x1'] = min(x1, x2)
        coords['x2'] = max(x1, x2)
        coords['y1'] = min(y1, y2)
        coords['y2'] = max(y1, y2)

        print("Selected ROI:", coords)

    rect_selector = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1],
        minspanx=2,
        minspany=2,
        spancoords='pixels',
        interactive=True
    )

    plt.show()

    if not coords:
        raise RuntimeError("No ROI selected.")

    return coords

def modality_score(sal):
    # sum over spatial dims, optionally also channels
    return sal.sum(dim=(1, 2, 3), keepdim=True)   # or (2,3) if already [1,1,H,W]

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
        return attr.abs().mean().item()


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
    attribution_method="saliency",   # "saliency" or "ig"
    ig_steps=64,
):
    x = x_init
    attributor = DiffusionAttributor(generator1, generator2)

    with torch.no_grad():
        unc_accum = torch.zeros_like(x_init)

    modality_scores = {
        "flair": 0.0,
        "t2": 0.0,
        "t1": 0.0,
    }

    saliency_maps = None
    saliency_steps = 0

    if return_saliency_maps:
        saliency_maps = {
            "flair": torch.zeros_like(cond1),
            "t2": torch.zeros_like(cond2),
            "t1": torch.zeros_like(cond3),
        }

    if step_stride is None:
        step_stride = 1 if track_modality_contrib else None

    for i in reversed(range(n_time)):
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

            forward_func = attributor.make_forward_func(
                x=x,
                t_time=t_time,
                latent_z=latent_z,
                roi_mask=roi_mask,
            )
            for method in attribution_method:
                if method == "saliency":
                    attr_method = Saliency(forward_func)
                    s1, s2, s3 = attr_method.attribute(
                        inputs=inputs,
                        abs=False,
                    )
                elif method == "ig":
                    attr_method = IntegratedGradients(forward_func)
                    i1, i2, i3 = attr_method.attribute(
                        inputs=inputs,
                        baselines=(
                            torch.zeros_like(cond1_step),
                            torch.zeros_like(cond2_step),
                            torch.zeros_like(cond3_step),
                        ),
                        n_steps=ig_steps,
                        method="gausslegendre",
                    )
                else:
                    raise ValueError(f"Unknown attribution_method: {attribution_method}")

            with torch.no_grad():
                if "saliency" in attribution_method:
                    modality_scores["flair"] += _reduce_attr(s1, positive_only=positive_only)
                    modality_scores["t2"] += _reduce_attr(s2, positive_only=positive_only)
                    modality_scores["t1"] += _reduce_attr(s3, positive_only=positive_only)
                elif "ig" in attribution_method:
                    modality_scores["flair"] += _reduce_attr(i1, positive_only=positive_only)
                    modality_scores["t2"] += _reduce_attr(i2, positive_only=positive_only)
                    modality_scores["t1"] += _reduce_attr(i3, positive_only=positive_only)

                if return_saliency_maps and "saliency" in attribution_method:
                    saliency_maps["flair"] += s1.detach()
                    saliency_maps["t2"] += s2.detach()
                    saliency_maps["t1"] += s3.detach()
                    saliency_steps += 1
                elif return_saliency_maps and "ig" in attribution_method:
                    saliency_maps["flair"] += i1.detach()
                    saliency_maps["t2"] += i2.detach()
                    saliency_maps["t1"] += i3.detach()
                    saliency_steps += 1

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

                x_new = sample_posterior_combine(
                    coefficients,
                    x_0_1[:, [0], :].detach(),
                    x_0_2[:, [0], :].detach(),
                    x,
                    t,
                )
                x = x_new.detach()

                del x_0_1, x_0_2, unc_map_t, x_new

        else:
            with torch.no_grad():
                x_0_1 = generator1(x, cond1, cond2, cond3, t_time, latent_z)
                x_0_2 = generator2(x, cond1, cond2, cond3, t_time, latent_z, x_0_1[:, [0], :])

                unc_map_t = torch.abs(x_0_1[:, [0], :] - x_0_2[:, [0], :])
                unc_accum = unc_accum + unc_map_t

                x_new = sample_posterior_combine(
                    coefficients,
                    x_0_1[:, [0], :].detach(),
                    x_0_2[:, [0], :].detach(),
                    x,
                    t,
                )
                x = x_new.detach()

                del x_0_1, x_0_2, unc_map_t, x_new

        del latent_z, t, t_time

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    unc = unc_accum / n_time

    if track_modality_contrib:
        total = modality_scores["flair"] + modality_scores["t2"] + modality_scores["t1"] + 1e-8
        modality_scores_norm = {
            "flair": modality_scores["flair"] / total,
            "t2": modality_scores["t2"] / total,
            "t1": modality_scores["t1"] / total,
        }

        if return_saliency_maps:
            if saliency_steps > 0:
                saliency_maps["flair"] /= saliency_steps
                saliency_maps["t2"] /= saliency_steps
                saliency_maps["t1"] /= saliency_steps

            if normalize_saliency_maps:
                for k in saliency_maps:
                    sm = saliency_maps[k]
                    # reduce small values (percentile threshold)
                    sm_np = sm.cpu().numpy()
                    thresh = np.percentile(sm_np, 95)  # top 5% strongest
                    sm = torch.where(sm < thresh, torch.zeros_like(sm), sm)
                    sm_min = sm.min()
                    sm_max = sm.max()
                    saliency_maps[k] = (sm - sm_min) / (sm_max - sm_min + 1e-8)

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
def load_image(image_path, size=(256, 256)):
    """
    Loads an image from the specified path, preprocesses it, and returns it as a tensor with shape (1, 1, 256, 256).
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size),
        torchvision.transforms.ToTensor()  # This will convert the image to a tensor of shape (1, 256, 256)
    ])
    
    img = Image.open(image_path).convert("L")  # Convert to grayscale ('L' mode for single channel)
    img_np = np.array(img)  # Convert to numpy array for min-max scaling

    
    # Apply IRM min-max pre-processing
    img_np = irm_min_max_preprocess(img_np)
    
    # Normalize to [-1, 1] by applying (data - 0.5) / 0.5
    img_np = (img_np - 0.5) / 0.5
    
    # Convert back to tensor and add batch and channel dimensions
    img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 256, 256)
    
    return img_tensor

if __name__ == "__main__":
    device='cuda:0'
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

    """ modalities = ['t1','t2','flair','t1ce']
    for modality in modalities:
        arr = np.load(rf"save_dir_path/{modality}/contrast.npy")
        # If shape is (1, 1, 256, 256)
        if arr.ndim == 4:
            arr = arr.squeeze()          # -> (256, 256)
       # If shape is (1, 1, 256)
        if arr.ndim == 3:
            arr = arr.squeeze()
        arr = Image.fromarray(arr[79, 0:256, 0:256])
        if modality == 't1ce':
            real_data = preprocess_image(arr).cuda()  # shape: [C, H, W]
        elif modality == 't1':
            x3 = preprocess_image(arr).cuda()  # shape: [C, H, W]
        elif modality == 't2':
            x2 = preprocess_image(arr).cuda()  # shape: [C, H, W]
        elif modality == 'flair':
            x1 = preprocess_image(arr).cuda()  # shape: [C, H, W] """

    x1_path = r'demo/sample_data/flair.jpg'
    x2_path = r'demo/sample_data/t2.jpg'
    x3_path = r'demo/sample_data/t1.jpg'
    real_data_path = r'demo/sample_data/t1ce.jpg'

    # Load images
    x1 = load_image(x1_path).cuda()
    x2 = load_image(x2_path).cuda()
    x3 = load_image(x3_path).cuda()
    real_data = load_image(real_data_path).cuda()

    x1 = torch.rot90(x1, k=-1, dims=(2, 3))
    x2 = torch.rot90(x2, k=-1, dims=(2, 3))
    x3 = torch.rot90(x3, k=-1, dims=(2, 3))
    real_data = torch.rot90(real_data, k=-1, dims=(2, 3))
    real_for_roi = ((x3.squeeze().detach().cpu().numpy() + 1.0) / 2.0) * 255.0
    roi = select_roi_interactive(real_for_roi)

    roi_mask = torch.zeros_like(real_data)
    roi_mask[:, :, roi['y1']:roi['y2'], roi['x1']:roi['x2']] = 1.0

    # make inputs differentiable

    sample_inputs = torch.cat((x1, x2, x3, real_data), axis=-1)  # Concatenate along the width

    # Squeeze the tensor to remove the batch and channel dimensions for visualization
    sample_inputs = sample_inputs.squeeze(0).squeeze(0)  # Shape: (256, 256, 5)

    # Plot the concatenated image
    plt.figure(figsize=(10, 10))
    plt.imshow(sample_inputs.cpu().numpy(), cmap='gray')  # Display in grayscale
    plt.axis('off')  # Hide axes
    plt.show()

    T = get_time_schedule(args, device)
    pos_coeff = Posterior_Coefficients(args, device)

    # Initialize noisy input
    x1_t = torch.randn_like(real_data)
    print('Prep 3: Done')

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
        attribution_method=["ig"],   # "saliency" or "ig"
        ig_steps=4,
    )

    print("\nRaw modality scores:")
    print("FLAIR:", round(modality_scores_raw["flair"],4))
    print("T2:   ", round(modality_scores_raw["t2"],4))
    print("T1:   ", round(modality_scores_raw["t1"],4))

    print("\nNormalized modality fractions:")
    print("FLAIR:", round(modality_scores_norm["flair"],4))
    print("T2:   ", round(modality_scores_norm["t2"],4))
    print("T1:   ", round(modality_scores_norm["t1"],4))

    unc = unc - unc.min()
    unc = unc / (unc.max() + 1e-8)

    # Normalize and save
    to_range_0_1 = lambda x: (x + 1.) / 2.
    fake_sample = to_range_0_1(fake_sample)

    fake_sample = fake_sample * 255.0
    fake_sample = fake_sample.squeeze(0).squeeze(0)  # Shape: (256, 256, 5)

    # Plot the concatenated image
    plt.figure(figsize=(10, 10))
    plt.imshow(fake_sample.cpu().numpy(), cmap='gray')  # Display in grayscale
    plt.axis('off')  # Hide axes
    plt.savefig("Generated Image.png", bbox_inches="tight", pad_inches=0)

    plt.figure(figsize=(10, 10))
    plt.imshow(sample_inputs.cpu().numpy(), cmap='gray')  # Display in grayscale
    plt.axis('off')  # Hide axes
    plt.figure(figsize=(8, 8))
    plt.imshow(fake_sample.cpu().numpy(), cmap='gray')
    plt.imshow(unc.squeeze(0).squeeze(0).cpu().numpy(), cmap='jet', alpha=0.35)  # overlay
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04, label='Uncertainty')
    plt.savefig("uncertainty_overlay.png", bbox_inches="tight", pad_inches=0)
    plt.axis('off')

    fig, axes = plt.subplots(1, 3, figsize=(12, 8))
    axes[0].imshow(x1.squeeze(0).squeeze(0).cpu().numpy(), cmap='gray')
    axes[0].imshow(saliency_maps["flair"].squeeze(0).squeeze(0).cpu().numpy(), cmap='hot', alpha=0.45)
    axes[0].set_title("FLAIR saliency")

    axes[1].imshow(x2.squeeze(0).squeeze(0).cpu().numpy(), cmap='gray')
    axes[1].imshow(saliency_maps["t2"].squeeze(0).squeeze(0).cpu().numpy(), cmap='hot', alpha=0.45)
    axes[1].set_title("T2 saliency")

    axes[2].imshow(x3.squeeze(0).squeeze(0).cpu().numpy(), cmap='gray')
    axes[2].imshow(saliency_maps["t1"].squeeze(0).squeeze(0).cpu().numpy(), cmap='hot', alpha=0.45)
    axes[2].set_title("T1 saliency")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("modality_captum_IntGrad.png", dpi=200, bbox_inches="tight")
    plt.show()