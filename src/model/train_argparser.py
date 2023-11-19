import argparse


class HyperparamArgs:
    """Process arguments from argparse into a model"""

    def __init__(self, namespace: argparse.Namespace):
        self.lr: float = namespace.lr
        self.epochs: int = namespace.epochs
        self.batch_size: int = namespace.batch_size
        self.gradient_acc: int = namespace.gradient_acc
        self.use_mixed_precision: bool = bool(namespace.use_mixed_precision)
        self.dropout: float = namespace.dropout
        self.in_jupyter: bool = bool(namespace.in_jupyter)
        self.from_pretrained: str = namespace.from_pretrained
        self.device: str = namespace.device
        self.save_to: str = namespace.save_to
        self.compile: bool = namespace.compile

    def __repr__(self):
        return f"""Hyperparams:
        lr: {self.lr}
        epochs: {self.epochs}
        batch_size: {self.batch_size}
        gradient_acc: {self.gradient_acc}
        use_mixed_precision: {self.use_mixed_precision}
        dropout: {self.dropout}
        in_jupyter: {self.in_jupyter}
        from_pretrained: {self.from_pretrained}
        device: {self.device}
        save_to: {self.save_to}
        """


def parse_arguments() -> HyperparamArgs:
    """Parse argument switches from command line"""

    parser = argparse.ArgumentParser(
        description="Train the transformer base-model on unstructured text."
    )
    parser.add_argument(
        "--lr",
        default=0.0003,
        type=float,
        help="Set beginning learning rate",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=10,
        type=int,
        help="Number of epochs (training iterations)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=32,
        type=int,
        help="Number of batches to process at once (per step)",
    )
    parser.add_argument(
        "-ga",
        "--gradient-acc",
        default=4,
        type=int,
        help="Number of gradient accumulation steps (simulate larger batch size by accumulating gradients per step)",
    )
    parser.add_argument(
        "-mp",
        "--use-mixed-precision",
        action="store_true",
        help="Use automatic precision to speed up training (only on CUDA-enabled GPUs)",
    )
    parser.add_argument(
        "-d",
        "--dropout",
        default=0.0,
        type=float,
        help="Dropout rate to randomly drop out weights to reduce overfitting",
    )
    parser.add_argument(
        "-j",
        "--in-jupyter",
        default=0,
        type=int,
        help="Set to true if running in Jupyter Notebook",
    )
    parser.add_argument(
        "-f",
        "--from-pretrained",
        default="model_cache",
        type=str,
        help="Pretrained file checkpoint to load",
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="Accelerator device to use (supports: cuda, cpu, mps, xla.)",
    )
    parser.add_argument(
        "--save-to",
        default="model_cache",
        type=str,
        help="File to save to",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model with torch dynamo",
    )

    return HyperparamArgs(parser.parse_args())
