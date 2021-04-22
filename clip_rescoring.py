import argparse
import torch
import numpy
import random
from model import Generator
import clip
from PIL import Image
from clip_analysis import analysis
import pickle



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
    parser.add_argument("--save_path", type=str, required=True, help="save file path")
    

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
'age11': ['high age', 'low age'],
'age12': ['senior', 'junior'],
'age13': ['an aged person', 'a young person'],
'age14': ['an adult', 'a juvenile'],
'age15': ['old', 'young'],
'age16': ['an old person', 'a young person'],
'age17': ['a teenager', 'a kid'],
'age18': ['an adult', 'a teenager'],
'age19': ['an old person', 'an adult'],
'age21': ['an elderly person', 'a young person'],

'wrinkle22': ['face with wrinkle', 'face without wrinkle'],

'skin0': ['a black person', 'a white person'],
'skin1': ['a black ', 'a white '],
'skin2': ['a black ', 'an asian '],
'skin3': ['a asian ', 'an white'],
'skin4': ['a person with light colored skin', 'a person with dark colored skin '],

'beard0': ['a beard person', 'no beard person'],
'beard1': ['a person with beard', 'a person without beard'],
'beard2': ['beard', 'no beard'],
'beard3': ['mustache', 'no mustache'],
'beard4': ['has mustache', 'has not mustache'],
'beard5': ['a person with mustache', 'a person without mustache'],
'beard6': ['much mustache', 'little mustache'],
'beard7': ['much beard', 'little beard'],

'haircolor0': ['blonde hair', 'black hair'],
'haircolor1': ['white hair', 'black hair'],
'haircolor1': ['gray hair', 'black hair'],
'haircolor2': ['person with blonde hair', 'person with black hair'],
'haircolor3': ['person with white hair', 'person with black hair'],
'haircolor5': ['hair is black', 'hair is brown'],
'haircolor6': ['a brown hair person', 'a black hair person'],
'haircolor7': ['a gray hair person', 'a black hair person'],
'haircolor8': ['face with blonde hair', 'face with black hair'],
'haircolor9': ['face with white hair', 'face with black hair'],
'haircolor10': ['a white hair face', 'a black hair face'],
'haircolor11': ['person with gray hair', 'person with black hair'],
'haircolor12': ['a face with gray hair', 'face with black hair'],

'noses0': ['a long nose', 'a short nose'],
'noses1': ['person with long nose', 'person with short nose'],
'noses2': ['the nose is long', 'the nose is short'],
'noses3': ['shape of nose is long', 'shape of nose is short'],
'noses4': ['a hooknose', 'a saddle nose'],
'noses5': ['person with a hooknose', 'person with a saddle nose'],
'noses6': ['person with a sharp nose', 'person with a flat nose'],
'noses7': ['face with a hooknose', 'face with a saddle nose'],
'noses8': ['face with a sharp nose', 'face with a flat nose'],
'noses9': ['a sharp nose', 'a flat nose'],

'lips0': ['red lips', 'gray lips'],
'lips1': ['a firm lip', 'an open lip'],
'lips2': ['a thin lip', 'a thick lip'],
'lips3': ['person with a red lip', 'person with a gray lip'],
'lips4': ['person with a firm lip', 'person with an open lip'],
'lips5': ['person with a thin lip', 'person with a thick lip'],
'lips6': ['a wide lip', 'a round lip'],
'lips7': ['face with a wide lip', 'face with a round lip'],
'lips8': ['face with a red lip', 'face with a gray lip'],
'lips9': ['face with a firm lip', 'face with an open lip'],
'lips10': ['face with a thin lip', 'face with a thick lip'],
'lips11': ['thin lips', 'thick lips'],
'lips12': ['lips without lipsticks', 'lips with lipsticks'],

'bangs0': ['bangs', 'no bangs'],
'bangs1': ['has bangs', 'has not bangs'],
'bangs2': ['a person with bangs', 'a person without bangs'],
'bangs3': ['left bangs', 'right bangs'],
'bangs4': ['side bangs', 'full bangs'],
'bangs5': ['a person with side bangs', 'a person with full bangs'],
'bangs6': ['a face with side bangs', 'a face with full bangs'],
'bangs7': ['a face with bangs', 'a face without bangs'],

'hats0': ['a face without hat', 'a person with hat'],
'hats1': ['a person without hat', 'a person with hat'],
'hats2': ['a short hat', 'a long hat'],
'hats3': ['no hat', 'a hat'],
'hats4': ['no cap', 'a cap'],
'hats5': ['a face without cap', 'a face with cap'],

'hairpin1': ['a person with a hairpin', 'a person without a hairpin'],

'hairstyle0': ['a person with long hair', 'a person with short hair'],
'hairstyle1': ['a face with long hair', 'a face with short hair'],
'hairstyle2': ['long hair', 'short hair'],
'hairstyle3': ['curly hair', 'straight hair'],
'hairstyle4': ['a person with curly hair', 'a person with straight hair'],
'hairstyle5': ['a face with curly hair', 'a face with straight hair'],
'hairstyle6': ['the hair is long ', 'the hair is short'],

'mouth0': ['an open mouth', 'a closed mouth'],
'mouth1': ['face with an open mouth', 'face with an closed mouth'],
'mouth2': ['an open mouth', 'not an open mouth'],
'mouth3': ['a grinning mouth', 'a closed mouth'],
'mouth4': ['mouth is open', 'mouth is closed'],
'mouth5': ['a large mouth', 'a small mouth'],
'mouth6': ['person with large mouth', 'person with small mouth'],
'mouth7': ['face with large mouth', 'face with small mouth'],
'mouth8':['the mouth is visible', 'the mouth is not visible'],
'mouth9':['the teeth is visible', 'the teeth is not visible'],
'mouth10':['a mouth with teeth', 'a mouth without teeth'],
'mouth11':['a person with teeth', 'a person without teeth'],
'mouth12':['teeth', 'no teeth'],
'mouth13': ['a person who smiles with teeth', 'a person who smiles without teeth '],
'mouth15': ['a person with a grinning mouth', 'a person with a closed mouth'],
'mouth16': ['grin with showing teeth', 'grin without showing teeth'],
'mouth17': ['a person with a grinning mouth', 'a person with a closed mouth'],
'mouth21': ['a slightly open mouth', 'a tight mouth'],
'mouth22': ['a closed mouth', 'a big open mouth'],


'face0': ['a long face', 'a short face'],
'face1': ['a wide face', 'a thin face'],
'face2': ['a round face', 'a thin face'],
'face3': ['a big face', 'a small face'],
'face4': ['person with long face', 'person with short face'],
'face5': ['person with round face', 'person with thin face'],
'face6': ['person with round cheek', 'person with thin cheek'],
'face7': ['a round cheek', 'a thin cheek'],

'facedirection1': ['face looking to the left', 'face looking to the front'],
'facedirection2': ['face looking to the left', 'face looking to the right'],
'facedirection3': ['face looking to the right', 'face looking to the left'],
'facedirection4': ['a face to the left', 'a face to the front'],
'facedirection5': ['a face upwards', 'a face downwards'],
'facedirection6': ['a face downwards', 'a face upwards'],

'lit1': ['a face with frontlit', 'a face with backlit'],
'lit2': ['frontlit', 'backlit'],
'lit3': ['a face with frontlight', 'a face with backlight'],
'lit4': ['frontlight', 'backlight'],
'lit5': ['a face rightlit', 'a face leftlit'],
'lit5': ['rightlight', 'leftlight'],

'eye0': ['a face with open eye', 'a face with closed eye'],
'eye1': ['a person with big eye', 'a person with small eye'],
'eye2': ['the eye is big', 'the eye is small'],
'eye3': ['the eye is huge', 'the eye is tiny'],
'eye4':['an eye looking left', 'an eye looking right'],
'eye5':['a person looking left', 'a person looking right'],
'eye6':['a face staring at left', 'a face staring at right'],
'eye7':['a face looking left', 'a face looking right'],
'eye8':['a person with closed eye', 'a person with open eye'],
'eyes9': ['a face with big eyes', 'a face with small eyes '],
'eyes10': ['blue eyes', 'black eyes '],
'eyes11': ['eyes looking right', 'eyes looking front'],
'eyes12': ['eyes looking down', 'eyes looking up'],
'eyes13': ['brown eyes', 'black eyes'],
'eyes14': ['little eyes', 'big eyes'],
'eyes15': ['big eyes', 'slender eyes'],

'cloth0':['color of clothes is dark', 'color of clothes is light'],
'cloth1':['dark clothes', 'light clothes'],
'cloth2':['color of cloth is dark', 'color of cloth is light'],
'cloth3':['light cloth', 'dark cloth'],
'cloth4':['a person with dark clothes', 'a person with light clothes'],
'cloth5':['a person with dark cloth', 'a person with light cloth'],
'cloth6':['a person with dark cloth', 'a person with shallow cloth'],

'light4':['light is strong', 'light is weak'],
'light5':['strong light', 'weak light'],
'light6':['shining light', 'dim light'],
'light7':['colorful background', 'black background'],
'light8':['RGB image', 'gray image'],
'light9':['sharp image', 'blur image'],
'light10': ['red light ', 'blue light'],
'light11': ['colored light ', 'normal light'],

'expression0':['smile', 'not smile'],
'expression1':['a smile person', 'not smile person'],
'expression2':['a face smiling', 'not smiling face'],
'expression3':['a person with smile', 'a person without smile'],
'expression4':['laugh', 'smile'],
'expression5':['big grins', 'small grins'],
'expression6':['a face with big grins', 'a face with small grins'],
'expression7':['a face with grins', 'a face without grins'],


'emotion1': ['placid', 'glad'],
'emotion2': ['glad', 'placid'],
'emotion3': ['expressionless', 'happy'],
'emotion4': ['happy', 'expressionless'],
'emotion5': ['expressionless', 'delighted'],
'emotion6': ['delighted', 'puzzled'],
'emotion7': ['excited', 'terrified'],
'emotion8': ['placid', 'surprised'],
'emotion9': ['astonished', 'happy'],
'emotion10': ['puzzled', 'excited'],
'emotion11': ['amazed', 'happy'],
'emotion12': ['excited', 'pleasant'],
'emotion13': ['excited', 'sad'],
'emotion14': ['shocked', 'glad'],
'emotion15': ['surprised', 'happy'],

'eyebrow0':['thick eyebrow', 'thin eyebrow'],
'eyebrow1':['person with thick eyebrow', 'person with thin eyebrow'],
'eyebrow2':['a face with thick eyebrow', 'a face with thin eyebrow'],
'eyebrow3':['the person\'s eyebrow is thick', 'the person\'s eyebrow is thin'],

'headdirection0':['looking left', 'looking right'],
'headdirection1':['looking up', 'looking down'],
'headdirection2':['a face to the left', 'a face to the right'],
'headdirection3':['a face looking left', 'a face looking right'],
'headdirection4':['a person looking left', 'a person looking right'],
'headdirection5':['looking over left', 'looking over right'],
'headdirection6':['a face looking over left', 'a face looking over right'],
'headdirection7':['a person looking over left', 'a person looking over right'],
'headdirection8':['a face looking up', 'a face looking down'],
'headdirection9':['a face up', 'a face down'],
'headdirection10':['a face upward', 'a face downward'],

'light1': ['a face with colored light on it ', 'a face with normal light on it '],
'light2': ['a face with orange light on it ', 'a face with normal light on it '],
'light3': ['a face with red light on it ', 'a face with blue light on it'],

'shape1': ['fat', 'thin'],
'shape2': ['a round face', 'a long face'],
'shape3': ['small', 'big'],
'shape4': ['a fat person', 'a thin person'],
'shape5': ['a person with round face', 'a person with long face'],
'shape6': ['a small person', 'a big person'],

'earings1': ['a person with earrings', 'a person without earrings'],
'earings2': ['a face with earrings', 'a face without earrings'],
'earings3': ['earrings', 'no earrings'],

'shirt color1': ['a white shirt', 'a black shirt'],
'shirt color2': ['a light shirt', 'a dark shirt'],
'shirt color3': ['a bright shirt', 'a dark shirt'],
'shirt color4': ['a colored shirt', 'a shirt without color'],
'shirt color5': ['a shirt', 'a red shirt'],
'shirt color6': ['a shirt', 'a green shirt'],
'shirt color7': ['a red shirt', 'a green shirt'],

'background1': ['a dark brown background', 'a lighted background'],
'background2': ['a white background', 'a dark background'],
'background3': ['a background of nature', 'a background of nothing'],
}
    print(semantic_attributes)

    save_file_name = args.save_path + "result_" + "i_" + str(index) + ".p"

    analysis(
      g, 
      eigvec[:, index].unsqueeze(0),
      semantic_attributes,
      number_of_samples = args.n_sample,
      number_of_variations = args.variant,
      variation_degrees = args.degree,
      save_file_name = save_file_name
    )
    
