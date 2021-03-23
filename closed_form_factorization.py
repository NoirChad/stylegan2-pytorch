import argparse

import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract factor/eigenvectors of latent spaces using closed form factorization"
    )

    parser.add_argument(
        "--out", type=str, default="factor.pt", help="name of the result factor file"
    )
    parser.add_argument("ckpt", type=str, help="name of the model checkpoint")
    parser.add_argument('--full_model', default=False, action='store_true')

    args = parser.parse_args()

    if args.full_model:
      state_dict = torch.load(args.ckpt).state_dict()
    else:
      state_dict = torch.load(args.ckpt)["g_ema"]

    modulate = {
        k: v
        for k, v in state_dict.items()
        if "modulation" in k and "to_rgbs" not in k and "weight" in k
    }

    weight_mat = []
    for k, v in modulate.items():
        weight_mat.append(v)

    W = torch.cat(weight_mat, 0)
    eigvec = torch.svd(W).V.to("cpu")

    torch.save({"ckpt": args.ckpt, "eigvec": eigvec}, args.out)

