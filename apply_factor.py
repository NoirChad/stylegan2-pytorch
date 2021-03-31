import argparse

import torch
import numpy
import random
from torchvision import utils

from model import Generator


if __name__ == "__main__":
    
    torch.manual_seed(1)
    numpy.random.seed(1)
    random.seed(1)
    
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Apply closed form factorization")

    parser.add_argument(
        "-i", "--index", type=int, default=0, help="index of eigenvector"
    )
    parser.add_argument(
        "-d",
        "--degree",
        type=float,
        default=5,
        help="scalar factors for moving latent vectors along eigenvector",
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
        "-n", "--n_sample", type=int, default=7, help="number of samples created"
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
    parser.add_argument(
        "factor",
        type=str,
        help="name of the closed form factorization result factor file",
    )
    parser.add_argument('--full_model', default=False, action='store_true')

    parser.add_argument("-c", "--col", type=int, default=None, help="col")
    parser.add_argument("-r", "--row", type=int, default=None, help="row")

    args = parser.parse_args()

    eigvec = torch.load(args.factor)["eigvec"].to(args.device)
    ckpt = torch.load(args.ckpt)
    if args.full_model:
        g = torch.load(args.ckpt).to(args.device)
    else:
        g = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, max_channel_size=args.max_channel_size
        ).to(args.device)
        checkpoint = torch.load(args.ckpt)

        g.load_state_dict(checkpoint["g_ema"])

    trunc = g.mean_latent(4096)

    latent = torch.randn(args.n_sample, 512, device=args.device)
    latent = g.get_latent(latent)

    direction = args.degree * eigvec[:, args.index].unsqueeze(0)

    if args.row:
      latent = latent[args.row]

    alpha = range(-5 // 2 + 1, 5 // 2 + 1)
    directional_results = []

    imgs = []

    if args.col:
        imgs = []
        i_range = range(5)[args.col:args.col + 1]
    else:
        imgs = []
        i_range = range(5)

    for i in i_range:
        target_latent = latent - alpha[i] * direction
        img, _ = g(
          [target_latent],
          truncation=args.truncation,
          truncation_latent=trunc,
          input_is_latent=True,
        )
        #img = resize_transoform(img)
        imgs += [img]
    final_image = torch.cat(imgs).unsqueeze(0)
    final_image = torch.transpose(final_image, 0, 1)
      
    directional_results += [final_image]

    if args.row:
      nrow = 1
    else:
      nrow = args.n_sample
    final_image = torch.cat(directional_results, 0)

    final_image = final_image.reshape(final_image.shape[0] * final_image.shape[1], final_image.shape[2], final_image.shape[3], final_image.shape[4])

    grid = utils.save_image(
        final_image,
        f"{args.out_prefix}_index-{args.index}_degree-{args.degree}.png",
        pad_value  = 1,
        normalize=True,
        range=(-1, 1),
        nrow=nrow,
    )
    
