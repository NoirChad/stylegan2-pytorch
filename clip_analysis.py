import clip
import torch
from torchvision import utils
from torchvision import transforms
from scipy import stats
import numpy as np

def logit(p):
    return np.log(p) - np.log(1 - p)

def regression(base, probs):
  ones = torch.ones(base.shape[0], dtype=torch.float32)
  base = torch.stack([ones, base], dim = 1)
  inv = torch.pinverse(base)
  result = inv @ logit(probs)
  return result[1]

def analysis(
  generator, 
  direction,
  semantic_attributes,
  number_of_samples = 16,
  number_of_variations = 15,
  variation_degrees = 1,
  truncation = 0.7,
  device = 'cuda',
  print_examples = False
):

  model, preprocess = clip.load("ViT-B/32", device=device)

  tokenized_text = {}
  for key, value in semantic_attributes.items():
    tokenized_text[key] = clip.tokenize(value).to(device)

  if number_of_variations % 2 == 1:
    alpha = range(-number_of_variations // 2 + 1, number_of_variations // 2 + 1)
  else:
    assert "number of variations must be odd"

  trunc = generator.mean_latent(4096)
  latent = torch.randn(number_of_samples, 512, device=device)
  latent = generator.get_latent(latent)

  
  latents = []
  for i in range(number_of_variations):
    latents.append((latent - variation_degrees * alpha[i] * direction).unsqueeze(1))
  latent_matrix = torch.cat(latents, dim = 1)



  resize_transoform_224 = transforms.Resize(224)
  resize_transoform_64 = transforms.Resize(64)
  to_pil_transoform = transforms.ToPILImage()
  to_tensor_transoform = transforms.ToTensor()


  slope = {}
  for key, value in semantic_attributes.items():
    slope[key] = torch.zeros(len(value), dtype=torch.float)

  example_images = []
  for i in range(number_of_samples):
    image, _ = generator(
            [latent_matrix[i]],
            truncation=truncation,
            truncation_latent=trunc,
            input_is_latent=True,
          )
    if print_examples and i < 10:
      example_images.append(resize_transoform_64(image))
    image = resize_transoform_224(image)
    for key, text in tokenized_text.items():
      logits_per_image, logits_per_text = model(image, text)
      probs = logits_per_image.softmax(dim=-1).to('cpu')
      result = regression(variation_degrees * torch.tensor(list(alpha), dtype=torch.float32), probs.float())
      slope[key] += result
    if i % 10 == 9:
      print(str(i+1) + "/" + str(number_of_samples))

  for key, value in semantic_attributes.items():
    slope[key] /= number_of_samples
    print(str(key) + " : " + str(value))
    print(slope[key])


  if print_examples:
    final_image = torch.cat(example_images)
    grid = utils.save_image(
          final_image,
          f"example.png",
          pad_value  = 1,
          normalize=True,
          range=(-1, 1),
          nrow=number_of_variations,
      )


