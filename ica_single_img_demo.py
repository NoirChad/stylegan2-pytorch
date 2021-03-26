import argparse

import torch
import random
from torchvision import utils
from sklearn.decomposition import FastICA
import numpy as np
import random

def indipendent_components_decomposition(W, n_components):
      fast_ica = FastICA(n_components=n_components)
      fast_ica.fit(W)
      W_ = fast_ica.components_
      norm = np.linalg.norm(W_, axis = 1).reshape(-1, n_components)
      W_nomralize = W_ / norm.T
      indipendent_components = torch.from_numpy(W_nomralize.T).float()
      return indipendent_components

from model import Generator


if __name__ == "__main__":
    
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Apply closed form factorization")

    parser.add_argument(
        "-d",
        "--degree",
        type=float,
        default=4,
        help="scalar factors for moving latent vectors along eigenvector",
    )
    parser.add_argument(
        "-n", "--number_of_component", type=int, default=8, help="index of eigenvector"
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help='channel multiplier factor. config-f = 2, else = 1',
    )
    parser.add_argument(
        "--latent",
        type=int,
        default=512,
        help="demension of the latent",
    )
    parser.add_argument(
        "--n_mlp",
        type=int,
        default=8,
        help="n_mlp",
    )
    parser.add_argument(
        "--max_channel_size",
        type=int,
        default=512,
        help="max channel size",
    )
    parser.add_argument("--ckpt", type=str, required=True, help="stylegan2 checkpoints")
    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "--truncation", type=float, default=0.7, help="truncation factor"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to run the model"
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="factor",
        help="filename prefix to result samples",
    )
    parser.add_argument('--full_model', default=False, action='store_true')
    parser.add_argument('--transpose', default=False, action='store_true')

    args = parser.parse_args()

    ckpt = torch.load(args.ckpt)

    if args.full_model:
        state_dict = ckpt.state_dict()
        g = ckpt.to(args.device)
    else:
        state_dict = ckpt["g_ema"]
        g = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, max_channel_size=args.max_channel_size
        ).to(args.device)
        g.load_state_dict(state_dict)

    modulate = {
        k: v
        for k, v in state_dict.items()
        if "modulation" in k and "to_rgbs" not in k and "weight" in k
    }

    weight_mat = []
    for k, v in modulate.items():
        weight_mat.append(v)

    W = torch.cat(weight_mat, 0)

    num_of_components = args.number_of_component

    np.random.seed(0)
    random.seed(0)
    components = indipendent_components_decomposition(W, num_of_components).to(args.device)

    trunc = g.mean_latent(128)

    latent = trunc.tile((num_of_components,1))

    direction = args.degree * components[:, 0:num_of_components].T

    img, _ = g(
        [latent],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )
    img1, _ = g(
        [latent + direction],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )
    img2, _ = g(
        [latent - direction],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )
    img3, _ = g(
        [latent + 2 * direction],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )
    img4, _ = g(
        [latent - 2 * direction],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )
    print(img.shape)

    final_image = torch.cat([img3.unsqueeze(0), img1.unsqueeze(0), img.unsqueeze(0), img2.unsqueeze(0), img4.unsqueeze(0)], 0)

    print(final_image.shape)

    if args.transpose:
      final_image = torch.transpose(final_image, 0, 1)
      nrow = 5
    else:
      nrow = num_of_components

    print(final_image.shape)

    final_image = final_image.reshape(final_image.shape[0] * final_image.shape[1], final_image.shape[2], final_image.shape[3], final_image.shape[4])

    print(final_image.shape)

    grid = utils.save_image(
        final_image,
        f"demo.png",
        normalize=True,
        range=(-1, 1),
        nrow=nrow,
    )
