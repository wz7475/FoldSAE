from lightning import Trainer

from rfdiffusion.sae.sae import SAE, LitSAE
from argparse import ArgumentParser
from rfdiffusion.sae.dataloader import ActivationsDataModule

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("activations_dir")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--dataset_name", default="block4_pair")
    args = parser.parse_args()
    activations_dir = args.activations_dir
    batch_size = args.batch_size
    dataset_name = args.dataset_name

    model = LitSAE(SAE(128, 128*3))
    datamodule = ActivationsDataModule(activations_dir, dataset_name, batch_size)
    trainer = Trainer(max_epochs=20)
    trainer.fit(model, datamodule=datamodule)
    test_result = trainer.test(model, datamodule=datamodule)
    print(f"Test result: {test_result}")
