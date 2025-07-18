from argparse import ArgumentParser

import torch

from src.probes.probes import SimpleNetwork
import matplotlib.pyplot as plt



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--weight_path", "-w", default="SAE_latents/e77r3qfh/checkpoints/epoch=4-step=720.ckpt")
    parser.add_argument("--sae_type", "-s", choices=["pair", "non_pair"], default="non_pair")
    parser.add_argument("--top_k", "-k", default=10, type=int)
    parser.add_argument("--device", "-d", default="cuda")
    parser.add_argument("--mode", choices=["plots", "save_coef"], default="plots")
    parser.add_argument("--coef_path", default="coef.pt")
    args = parser.parse_args()
    weights_path = args.weight_path
    k = args.top_k
    device = args.device
    sae_type = args.sae_type
    mode = args.mode
    coef_path = args.coef_path
    model = SimpleNetwork.load_from_checkpoint(weights_path)
    weights = list(model.model.parameters())[0].data.to(device)
    w = weights.squeeze()  # [4200]
    abs_w = torch.abs(w)
    if mode == "save_coef":
        top_k = torch.topk(abs_w, k).indices.to(device)
        print(top_k)
        torch.save(top_k, coef_path)
    elif mode == "plots":

        sorted_w, _ = torch.sort(w, descending=True)  # sorted magnitudes

        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        axes[0][0].boxplot(sorted_w.tolist())
        axes[0][0].set_ylabel("value of coefficients")
        axes[0][0].set_xlabel("")
        axes[0][0].set_title("box plot of coefficients values")

        axes[0][1].plot(sorted_w.tolist())
        axes[0][1].set_ylabel("sorted coefficients values")
        axes[0][1].set_xlabel("idx of feature")
        axes[0][1].set_title("plot of coefficients values")


        axes[1][0].plot(sorted_w.tolist()[-20:])
        axes[1][0].set_title("sorted coefficients values, lowest 20")
        axes[1][0].set_xlabel("idx of feature")
        axes[1][0].set_xticklabels(list(range(sorted_w.shape[0]-20, sorted_w.shape[0])))

        axes[1][1].plot(sorted_w.tolist()[:20])
        axes[1][1].set_title("sorted coefficients values, highest 20")
        axes[1][1].set_xlabel("idx of feature")
        # plt.tight_layout()

        fig.suptitle(f"{sae_type}", fontsize=20)
        fig.tight_layout()
        plt.show()
