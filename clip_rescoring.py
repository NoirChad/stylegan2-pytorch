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
      'gender1': ['a male', 'a female'],
      'gender2': ['a male person', 'a female person'],
      'gender3': ['male', 'female'],
      'gender4': ['male person', 'female person'],
      'gender5': ['a man', 'a woman'],
      'gender6': ['a man\'s face', 'a woman\'s face'],
      'gender7': ['a male\'s face', 'a female\'s face'],
      'gender8': ['face of a man', 'face of a woman'],
      'gender9': ['face of a male', 'face of a female'],
      'gender10': ['a face of a man', 'a face of a woman'],
      'gender11': ['a face of a male', 'a face of a female'],
      'gender12': ['a guy', 'a lady'],
      'gender13': ['a gentleman', 'a lady'],
      'gender14': ['a fellow', 'a lady'],
      'gender15': ['a sir', 'a madam'],
      'gender16': ['a face of a guy', 'a face of a lady'],
      'gender17': ['a face of a gentleman', 'a face of a lady'],
      'gender18': ['a face of a sir', 'a face of a madam'],
      'gender19': ['a face of a fellow', 'a face of a lady'],
      'gender20': ['a picture of a man', 'a picture of a woman'],
      'gender21': ['a picture of a male', 'a picture of a female'],
      'gender22': ['male people', 'female people'],
      'gender23': ['a boy', 'a girl'],

      'glass1': ['a person wearing glasses', 'a person not wearing glasses'],
      'glass2': ['a person with glasses', 'a person without glasses'],
      'glass3': ['a person wearing glasses', 'a person'],
      'glass4': ['a person with glasses', 'a person'],
      'glass5': ['a face wearing glasses', 'a face not wearing glasses'],
      'glass6': ['a face wearing glasses', 'a face without wearing glasses'],
      'glass7': ['a face with glasses', 'a face without glasses'],
      'glass8': ['glasses', 'no glasses'],
      'glass9': ['person with eyeglasses', 'person without eyeglasses'],
      'glass10': ['a person with eyeglasses', 'a person without eyeglasses'],

      'age1': ['a person with high age', 'a person with low age'],
      'age2': ['an old person', 'a young person'],
      'age3': ['a face of an old person', 'a face of a young person'],
      'age4': ['a person with greater age', 'a person with lower age'],
      'age5': ['an older person', 'a younger person'],
      'age6': ['a older human', 'a younger human'],
      'age7': ['an aged person', 'a adolescent person'],
      'age8': ['a senior', 'a junior'],
      'age9': ['a grown-up', 'a juvenile'],
      'age10': ['young', 'old'],
      'age11': ['low age', 'high age'],
      'age12': ['senior', 'junior'],

      'skin0':['a black person', 'a white person'],
      'skin1':['a black person', 'an asian person'],
      'skin2':['a black ', 'a white '],
      'skin3':['a black ', 'an asian '],
      'skin4':['a white person', 'a black person'],
      'skin5':['a white person', 'an asian person'],
      'skin6':['a white ', 'a black '],
      'skin7':['a white ', 'an asian '],
      'skin8':['an asian person', 'a black person'],
      'skin9':['an asian person', 'a white person'],
      'skin10':['an asian ', 'a black '],
      'skin11':['an asian ', 'a white '],

      'beard0':['a beard person', 'no beard person'],
      'beard1':['a person with beard', 'a person without beard'],
      'beard2':['beard', 'no beard'],
      'beard3':['mustache', 'no mustache'],
      'beard4':['has mustache', 'has not mustache'],
      'beard5':['a person with mustache', 'a person without mustache'],
      'beard6':['little mustache', 'much mustache'],
      'beard7':['little beard', 'much beard'],

      'haircolor0':['blonde hair', 'black hair'],
      'haircolor1':['white hair', 'black hair'],
      'haircolor2':['person with blonde hair', 'person with black hair'],
      'haircolor3':['person with white hair', 'person with black hair'],
      'haircolor4':['black hair', 'brown hair'],
      'haircolor5':['hair is black', 'hair is brown'],
      'haircolor6':['a brown hair person', 'a black hair person'],
      'haircolor7':['a white hair person', 'a black hair person'],
      'haircolor8':['face with blonde hair', 'face with black hair'],
      'haircolor9':['face with white hair', 'face with black hair'],
      'haircolor10':['a brown hair face', 'a black hair face'],
      'haircolor11':['a white hair face', 'a black hair face'],

      'noses0':['a long nose', 'a short nose'],
      'noses1':['person with long nose', 'person with short nose'],
      'noses2':['the nose is long', 'the nose is short'],
      'noses3':['shape of nose is long', 'shape of nose is short'],
      'noses4':['a hooknose', 'a saddle nose'],
      'noses5':['person with a hooknose', 'person with a saddle nose'],
      'noses6':['person with a sharp nose', 'person with a flat nose'],
      'noses7':['face with a hooknose', 'face with a saddle nose'],
      'noses8':['face with a sharp nose', 'face with a flat nose'],
      'noses9':['a sharp nose', 'a flat nose'],

      'lips0':['red lips', 'gray lips'],
      'lips1':['a firm lip', 'an open lip'],
      'lips2':['a thin lip', 'a thick lip'],
      'lips3':['person with a red lip', 'person with a gray lip'],
      'lips4':['person with a firm lip', 'person with an open lip'],
      'lips5':['person with a thin lip', 'person with a thick lip'],
      'lips6':['a wide lip', 'a round lip'],
      'lips7':['face with a wide lip', 'face with a round lip'],
      'lips8':['face with a red lip', 'face with a gray lip'],
      'lips9':['face with a firm lip', 'face with an open lip'],
      'lips10':['face with a thin lip', 'face with a thick lip'],

      'bangs0':['bangs', 'no bangs'],
      'bangs1':['has bangs', 'has not bangs'],
      'bangs2':['a person with bangs', 'a person without bangs'],
      'bangs3':['left bangs', 'right bangs'],
      'bangs4':['side bangs', 'full bangs'],
      'bangs5':['a person with side bangs', 'a person with full bangs'],
      'bangs6':['a face with side bangs', 'a face with full bangs'],
      'bangs7':['a face with bangs', 'a face without bangs'],

      'hats0':['a face without hat', 'a person with hat'],
      'hats1':['a person without hat', 'a person with hat'],
      'hats2':['a short hat', 'a long hat'],
      'hats3':['no hat', 'a hat'],
      'hats4':['no cap', 'a cap'],
      'hats5':['a face without cap', 'a face with cap'],

      'hairstyle0':['a person with long hair', 'a person with short hair'],
      'hairstyle1':['a face with long hair', 'a face with short hair'],
      'hairstyle2':['long hair', 'short hair'],
      'hairstyle3':['curly hair', 'straight hair'],
      'hairstyle4':['a person with curly hair', 'a person with straight hair'],
      'hairstyle5':['a face with curly hair', 'a face with straight hair'],
      'hairstyle6':['the hair is long ', 'the hair is short'],

      'age0':['young', 'old'],
      'age1':['a young face', 'an old face'],
      'age2':['a young person', 'a old person'],
      'age3':['a kid', 'a teenager'],
      'age4':['a teenager', 'an old person'],
      'age5':['face with wrinkle', 'face without wrinkle'],
      'age6':['an elderly person', 'a young person'],

      'mouse0':['an open mouth', 'a closed mouth'],
      'mouse1':['face with an open mouth', 'face with an closed mouth'],
      'mouse2':['an open mouth', 'not an open mouth'],
      'mouse3':['person with open mouse', 'person without open mouse'],
      'mouse4':['mouth is open', 'mouth is closed'],
      'mouse5':['a large mouth', 'a small mouth'],
      'mouse6':['person with large mouth', 'person with small mouth'],
      'mouse7':['face with large mouth', 'face with small mouth'],

      'face0':['a long face', 'a short face'],
      'face1':['a wide face', 'a thin face'],
      'face2':['a round face', 'a thin face'],
      'face3':['a big face', 'a small face'],
      'face4':['person with long face', 'person with short face'],
      'face5':['person with round face', 'person with thin face'],
      'face6':['person with round cheek', 'person with thin cheek'],
      'face7':['a round cheek', 'a thin cheek'],

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
    
