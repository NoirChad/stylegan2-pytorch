import argparse

import torch
import random
from torchvision import utils
from torchvision import transforms
from sklearn.decomposition import FastICA
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont

def indipendent_components_decomposition(W, n_components):
      fast_ica = FastICA(n_components=n_components)
      fast_ica.fit(W)
      W_ = fast_ica.components_
      norm = np.linalg.norm(W_, axis = 1).reshape(-1, n_components)
      W_nomralize = W_ / norm.T
      indipendent_components = torch.from_numpy(W_nomralize.T).float()
      return indipendent_components

from model import Generator


def ica_single_img(
        ckpt,
        degree = 5, 
        number_of_component = 8, 
        channel_multiplier = 2, 
        latent = 512,
        n_mlp = 8,
        max_channel_size = 512,
        size = 256,
        truncation = 0.7,
        device = 'cuda',
        full_model = False,
        initial_latent = None,
        resolution = 64,
        start_component = 0,
        end_component = None,
        num_of_columns = 5,
        col = None,
        row = None,
        no_index = False,
        seed = None,
        directions = None
    ):

    print("Loading checkpoints...")
    ckpt = torch.load(ckpt)
    if full_model:
        state_dict = ckpt.state_dict()
        g = ckpt.to(device)
    else:
        state_dict = ckpt["g_ema"]
        g = Generator(
            size, latent, n_mlp, channel_multiplier=channel_multiplier, max_channel_size=max_channel_size
        ).to(device)
        g.load_state_dict(state_dict)

    

    if directions is not None:
      components = directions
      num_of_components = directions.shape[1]
    else:

      modulate = {
        k: v
        for k, v in state_dict.items()
        if "modulation" in k and "to_rgbs" not in k and "weight" in k
      }

      weight_mat = []
      for k, v in modulate.items():
          weight_mat.append(v)

      W = torch.cat(weight_mat, 0)

      num_of_components = number_of_component

      torch.manual_seed(0)
      np.random.seed(0)
      random.seed(0)
      
      print("Start semantic factorizing...")
      components = indipendent_components_decomposition(W, num_of_components).to(device)
      print("Semantic factorizing finished!")

    print("Generating images..")

    if seed:
      torch.manual_seed(seed)
      np.random.seed(seed)
      random.seed(seed)
      trunc = g.mean_latent(4)
    else:
      trunc = g.mean_latent(4096)

    
    w_plus = False

    if initial_latent:
      proj = torch.load(initial_latent)
      key = list(proj.keys())
      latent = proj[key[0]]['latent'].detach().to(device)
      #print(proj[key[0]]['noise'])
      noise = proj[key[0]]['noise']
      if len(list(latent.shape)) == 2:
        w_plus = True
    else:
      latent = trunc
      noise = None

    alpha = range(-num_of_columns // 2 + 1, num_of_columns // 2 + 1)
    resize_transoform = transforms.Resize(resolution)
    to_pil_transoform = transforms.ToPILImage()
    to_tensor_transoform = transforms.ToTensor()

    directional_results = []

    if row:
      d_range = range(num_of_components)[row:row + 1]
    else:
      d_range = range(num_of_components)[start_component:end_component]

    for d in d_range:

      txt = Image.new("RGB", (48, 48), (255,255,255))
      draw = ImageDraw.Draw(txt)
      draw.text((0, 20), "i = " + str(d), fill=(0,0,0))
      txt = txt.resize((resolution, resolution))
      txt = to_tensor_transoform(txt).to(device).unsqueeze(0)
      
      imgs = [txt]

      direction = degree * components[:, d].T

      if col:
        imgs = []
        i_range = range(num_of_components)[col:col + 1]
      else:
        imgs = [txt]
        i_range = range(num_of_columns)

      if no_index:
        imgs = []

      for i in i_range:
        if w_plus:
          target_latent = torch.unsqueeze(latent, 0).clone()
          target_latent[0] = target_latent[0] + alpha[i] * direction
        else:
          target_latent = latent + alpha[i] * direction
        img, _ = g(
          [target_latent],
          input_is_latent=True,
          noise = noise
        )
        img = resize_transoform(img)
        imgs += [img]

      final_image = torch.cat(imgs).unsqueeze(0)
      final_image = torch.transpose(final_image, 0, 1)
      
      directional_results += [final_image]

    if col:
      nrow = 1
    else:
      nrow = num_of_columns + 1
    final_image = torch.cat(directional_results, 0)

    final_image = final_image.reshape(final_image.shape[0] * final_image.shape[1], final_image.shape[2], final_image.shape[3], final_image.shape[4])

    grid = utils.make_grid(
        final_image,
        pad_value  = 1,
        normalize=True,
        range=(-1, 1),
        nrow=nrow,
    )

    return to_pil_transoform(grid)
  

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
    parser.add_argument('--full_model', default=False, action='store_true')
    parser.add_argument("--initial_latent", type=str, required=False, default=None)
    parser.add_argument("--resolution", type=int, default=64, help="resolution")
    parser.add_argument("--start_component", type=int, default=0, help="start_component")
    parser.add_argument("--end_component", type=int, default=None, help="end_component")
    parser.add_argument("--num_of_columns", type=int, default=5, help="num_of_columns")
    parser.add_argument("--col", type=int, default=None, help="column")
    parser.add_argument("--row", type=int, default=None, help="row")
    parser.add_argument('--no_index', default=False, action='store_true')
    parser.add_argument("--random_seed", type=int, default=None, help="random seed")
    parser.add_argument("--factor", type=str, default=None, help="factor")

    args = parser.parse_args()

    if args.factor:
      directions = torch.load(args.factor)["eigvec"].to(args.device)
    else:
      directions = None

    grid = ica_single_img(
      args.ckpt,
      degree = args.degree,
      number_of_component = args.number_of_component,
      channel_multiplier = args.channel_multiplier,
      latent = args.latent,
      n_mlp = args.n_mlp,
      max_channel_size = args.max_channel_size,
      size = args.size,
      truncation = args.truncation,
      full_model = args.full_model,
      initial_latent = args.initial_latent,
      resolution = args.resolution,
      start_component = args.start_component,
      end_component = args.end_component,
      num_of_columns = args.num_of_columns,
      col = args.col,
      row = args.row,
      no_index = args.no_index,
      seed = args.random_seed,
      directions = directions
      )

    grid.save("demo.png")
