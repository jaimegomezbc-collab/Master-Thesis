from PIL import Image
import torch.nn as nn
import numpy as np
import os, json
from backbones.ncsnpp_generator_adagn_feat import NCSNpp
from backbones.ncsnpp_generator_adagn_feat import NCSNpp_adaptive
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from scipy import ndimage

import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
from monai.networks.nets import DenseNet121

def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB') 
        
def get_input_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])       
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])    

    return transf

def get_input_tensors(img):
    transf = get_input_transform()
    # unsqeeze converts single image to batch of 1
    return transf(img).unsqueeze(0)

model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=2)
model.load_state_dict(torch.load("/cs/student/project_msc/2025/aibh/jgomezbe/Master-Thesis/best_metric_model.pth", weights_only=True))
model.eval
def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])    

    return transf

def get_preprocess_transform():
    transf = transforms.Compose([
        transforms.ToTensor(),
    ])    

    return transf    

pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()

def load_checkpoint(checkpoint_dir, netG, name_of_network, device='cuda:0'):
    checkpoint_file = checkpoint_dir.format(name_of_network)

    checkpoint = torch.load(checkpoint_file, map_location=device)
    ckpt = checkpoint

    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    netG.load_state_dict(ckpt, strict=False)
    netG.eval()

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

def sample_from_model(coefficients, generator1, cond1, generator2, cond2, cond3, n_time, x_init, T, opt):
    x = x_init

    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)

            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)  # .to(x.device)

            x_0_1 = generator1(x, cond1, cond2, cond3, t_time, latent_z)
            x_0_2 = generator2(x, cond1, cond2, cond3, t_time, latent_z, x_0_1[:, [0], :])

            x_new = sample_posterior_combine(coefficients, x_0_1[:, [0], :], x_0_2[:, [0], :], x, t)

            x = x_new.detach()

    return x

def normalize(image):
    """Basic min max scaler."""
    min_ = np.min(image)
    max_ = np.max(image)
    scale = max_ - min_
    image = (image - min_) / scale
    return image

def irm_min_max_preprocess(image, low_perc=1, high_perc=99):
    """Main pre-processing function for removing outliers and scaling."""
    non_zeros = image > 0
    low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
    image = np.clip(image, low, high)
    image = normalize(image)
    return image

def load_image(image_path, size=(256, 256)):
    """
    Loads an image from the specified path, preprocesses it, and returns it as a tensor with shape (1, 1, 256, 256).
    """
    
    img = Image.open(image_path).convert("L")  # Convert to grayscale ('L' mode for single channel)
    img = img.resize(size, Image.BILINEAR)
    img_np = np.array(img)  # Convert to numpy array for min-max scaling

    
    # Apply IRM min-max pre-processing
    img_np = irm_min_max_preprocess(img_np)
    
    # Normalize to [-1, 1] by applying (data - 0.5) / 0.5
    img_np = (img_np - 0.5) / 0.5
    
    # Convert back to tensor and add batch and channel dimensions
    img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 256, 256)
    
    return img_tensor

def generate_synthetic_image(flair, t2, t1):
    # Assuming flair, t2, t1 are PIL images
    device='cuda:0'
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

    gen_diffusive_1 = NCSNpp(args).to(device)
    gen_diffusive_2 = NCSNpp_adaptive(args).to(device)

    # Load checkpoints
    load_checkpoint(r'MU-Diff_Model_Weights/brats/t1ce/gen_diffusive_1.pth', gen_diffusive_1, 'gen_diffusive_1', device=device)
    load_checkpoint(r'MU-Diff_Model_Weights/brats/t1ce/gen_diffusive_2.pth', gen_diffusive_2, 'gen_diffusive_2', device=device)

    flair = torch.from_numpy(flair).unsqueeze(0).unsqueeze(0).cuda()
    t2 = torch.from_numpy(t2).unsqueeze(0).unsqueeze(0).cuda()
    t1 = torch.from_numpy(t1).unsqueeze(0).unsqueeze(0).cuda()

    # Load images
    x1=torch.rot90(flair, k=-1, dims=(2, 3))
    x2=torch.rot90(t2, k=-1, dims=(2, 3))
    x3=torch.rot90(t1, k=-1, dims=(2, 3))

    T = get_time_schedule(args, device)
    pos_coeff = Posterior_Coefficients(args, device)

    # Initialize noisy input
    x1_t = torch.randn_like(x1)
    fake_sample = sample_from_model(pos_coeff, gen_diffusive_1, x1, gen_diffusive_2, x2, x3,
                                    args.num_timesteps, x1_t, T, args)
    
    # Normalize and save
    to_range_0_1 = lambda x: (x + 1.) / 2.
    fake_sample = to_range_0_1(fake_sample)
    

    fake_sample = fake_sample*255.0
    fake_sample = fake_sample.squeeze(0).squeeze(0)  # Shape: (256, 256, 5)
    
    return fake_sample.detach().cpu().numpy()

def batch_predict(images, n_modes=1):
    synthetic_images = []
    if n_modes != 1:
        for i in range(images.shape[0]):
            print(i)
            synthetic_image = generate_synthetic_image(images[i][0], images[i][1], images[i][2])
            synthetic_images.append(synthetic_image)
    batch = torch.stack(tuple(preprocess_transform(i) for i in synthetic_images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

from Lime import lime_image

x1_path = '/cs/student/project_msc/2025/aibh/jgomezbe/UCSD-PTGBM/middle_slices/flair/UCSD-PTGBM-0002_01_slice131.npy'
x2_path = "/cs/student/project_msc/2025/aibh/jgomezbe/UCSD-PTGBM/middle_slices/t2/UCSD-PTGBM-0002_01_slice131.npy"
x3_path = "/cs/student/project_msc/2025/aibh/jgomezbe/UCSD-PTGBM/middle_slices/t1/UCSD-PTGBM-0002_01_slice131.npy"
img_name = os.path.basename(x1_path).split('/')[-1]
img_name = img_name[:-4]

# Load images
flair = np.load(x1_path)
t2 = np.load(x2_path)
t1 = np.load(x3_path)
plt.figure()
plt.imshow(t1,cmap='gray')
plt.title('FLAIR')  # Add title
plt.show()
print(t1.max())
img = np.stack([flair, t2, t1], axis=0)
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(img,
                                         batch_predict, # classification function
                                         top_labels=2,
                                         hide_color=0, 
                                         num_samples=1000,
                                         n_modes = 3,
                                         img_name= img_name)
from skimage.segmentation import mark_boundaries
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False)
plt.figure(figsize=(10, 10))
plt.subplot(131)
img_boundry1 = mark_boundaries(temp[0,:,:,]/255.0, mask[0,:,:])
plt.imshow(temp[0,:,:],cmap='gray', vmax = 2800)
plt.imshow(img_boundry1,alpha=0.5)  # Display in grayscale
plt.title('FLAIR')  # Add title
plt.axis('off')
plt.subplot(132)
img_boundry2 = mark_boundaries(temp[1,:,:]/255.0, mask[1,:,:])
plt.imshow(temp[1,:,:],cmap='gray', vmax = 7800)
plt.imshow(img_boundry2,alpha=0.5)
plt.title('T2')  # Add title
plt.axis('off')
plt.subplot(133)
img_boundry3 = mark_boundaries(temp[2,:,:]/255.0, mask[2,:,:])
plt.imshow(temp[2,:,:],cmap='gray', vmax = 4200)
plt.imshow(img_boundry3, alpha=0.5)
plt.title('T1')  # Add title
plt.axis('off')
plt.savefig(f"LIME_explanation_{img_name}.png")
plt.show()