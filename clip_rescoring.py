import argparse
import torch
import numpy
import random
from model import Generator
import clip
from PIL import Image
from clip_analysis import analysis



if __name__ == "__main__":
    torch.manual_seed(0)
    numpy.random.seed(0)
    random.seed(0)


    
    torch.set_grad_enabled(False)

    
    parser = argparse.ArgumentParser(description="Apply closed form factorization")
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
        "-n", "--n_sample", type=int, default=8, help="number of samples created"
    )
    parser.add_argument(
        "-v", "--variant", type=int, default=8, help="number of samples created"
    )
    parser.add_argument(
        "--truncation", type=float, default=0.7, help="truncation factor"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to run the model"
    )
    parser.add_argument(
        "factor",
        type=str,
        help="name of the closed form factorization result factor file",
    )
    parser.add_argument(
        "-d",
        "--degree",
        type=float,
        default=1,
        help="scalar factors for moving latent vectors along eigenvector",
    )
    parser.add_argument(
        "-i", "--index", type=int, default=0, help="the index"
    )

    

    args = parser.parse_args()

    index = args.index

    eigvec = torch.load(args.factor)["eigvec"].to(args.device)
    g = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, max_channel_size=args.max_channel_size
    ).to(args.device)
    checkpoint = torch.load(args.ckpt)
    g.load_state_dict(checkpoint["g_ema"])


    semantic_attributes = {
      'gender': ['a male', 'a female'],
      'age' : ['a younger person', 'an older person'],
      'skin color': ['a black person', 'a white person'],
      'hair color': ['a person with black hair', 'a person with blonde hair', 'a person with white hair'],

      'noses': ['a person with long noses', 'a person with short noses'],
      'beard': ['a person with beard', 'a person without beard'],
      'lips': ['a person with red lips', 'a person without red lips'],

      'bangs': ['a person with bangs', 'a person without bangs'],
      'glasses': ['a person with glasses', 'a person without glasses'],
      'hats': ['a person with hats', 'a person without hats'],

      'cloth_1': ['a person in green clothes', 'a person in red clothes', 'a person in blue clothes'],
      'cloth_2': ['a person dressed warm', 'a person dressed cool'],

      'mouth open': ['mouth open', 'mouth closed'],
      'eyes open': ['eyes open', 'eyes closed'],

      'image_background_1': ['a picture with white background', 'a picture with dark background'],
      'image_background_2': ['a picture with red background', 'a picture with green background', 'a picture with blue background'],
    }
    print(semantic_attributes)

    analysis(
      g, 
      eigvec[:, index].unsqueeze(0),
      semantic_attributes,
      number_of_samples = args.n_sample,
      number_of_variations = args.variant,
      variation_degrees = args.degree
    )
    
