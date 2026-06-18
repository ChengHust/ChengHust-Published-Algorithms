from __future__ import annotations

import argparse
from types import SimpleNamespace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal OPT + MOZO demo.")
    parser.add_argument("--model_name", default="facebook/opt-125m", help="HuggingFace OPT model id.")
    parser.add_argument("--dataset_name", default="wikitext", help="HuggingFace dataset name.")
    parser.add_argument("--dataset_config", default="wikitext-2-raw-v1", help="Dataset config/subset.")
    parser.add_argument("--split", default="train", help="Dataset split.")
    parser.add_argument("--max_samples", type=int, default=64, help="Number of text samples to load.")
    parser.add_argument("--max_length", type=int, default=128, help="Token sequence length.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=3, help="MOZO update steps.")
    parser.add_argument("--eval_batches", type=int, default=1, help="Batches used per objective evaluation.")
    parser.add_argument("--population_size", type=int, default=2)
    parser.add_argument("--zo_eps", type=float, default=1e-3)
    parser.add_argument("--mozo_step_size", type=float, default=1e-5)
    parser.add_argument(
        "--trainable",
        choices=["all", "final_attention", "lm_head"],
        default="all",
        help="Which OPT parameters MOZO is allowed to optimize. The paper setting is all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mozo_gpu import current_trainable_individual, mozo_step

    class TextDataset(Dataset):
        def __init__(self, encodings, pad_token_id):
            self.input_ids = encodings["input_ids"]
            self.attention_mask = encodings["attention_mask"]
            self.labels = self.input_ids.clone()
            self.labels[self.input_ids == pad_token_id] = -100

        def __len__(self):
            return self.input_ids.size(0)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.labels[idx],
            }

    class MinimalMOZOTrainer:
        def __init__(self, model, dataloader, trainer_args):
            self.model = model
            self.train_batch_dataloader = dataloader
            self.args = trainer_args
            self.weights = None
            self.mozo_pop = None

        def load_individual(self, model, individual):
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in individual:
                        param.copy_(individual[name].to(device=param.device, dtype=param.dtype))

        def evaluate_individual(self, model, individual, dataloader):
            self.load_individual(model, individual)
            model.eval()

            total_loss = 0.0
            total_batches = 0
            for batch in dataloader:
                batch = {key: value.to(model.device) for key, value in batch.items()}
                with torch.no_grad():
                    output = model(**batch)
                total_loss += float(output.loss.detach().cpu())
                total_batches += 1
                if total_batches >= self.args.eval_batches:
                    break

            trainable = current_trainable_individual(model)
            l2_norm = torch.sqrt(
                sum(torch.sum(value.float() ** 2) for value in trainable.values())
            ).item()
            return [total_loss / max(total_batches, 1), l2_norm]

    def build_dataloader(tokenizer):
        dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
        dataset = dataset.shuffle(seed=0)
        texts = [row["text"] for row in dataset if row.get("text", "").strip()]
        texts = texts[: args.max_samples]
        if not texts:
            raise RuntimeError("No non-empty text samples were found in the selected dataset split.")

        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_tensors="pt",
        )
        text_dataset = TextDataset(encodings, tokenizer.pad_token_id)
        return DataLoader(text_dataset, batch_size=args.batch_size, shuffle=True)

    def select_trainable_parameters(model, mode):
        for param in model.parameters():
            param.requires_grad = False

        if mode == "all":
            for param in model.parameters():
                param.requires_grad = True
            return

        if mode == "lm_head":
            output_embeddings = model.get_output_embeddings()
            if output_embeddings is None:
                raise RuntimeError("The selected model does not expose output embeddings.")
            for param in output_embeddings.parameters():
                param.requires_grad = True
            return

        num_layers = getattr(model.config, "num_hidden_layers", None)
        if num_layers is None:
            raise RuntimeError("Cannot infer the final OPT layer from model.config.num_hidden_layers.")

        final_prefix = f"model.decoder.layers.{num_layers - 1}.self_attn"
        trainable_count = 0
        for name, param in model.named_parameters():
            if name.startswith(final_prefix):
                param.requires_grad = True
                trainable_count += param.numel()

        if trainable_count == 0:
            raise RuntimeError(f"No parameters matched {final_prefix!r}; check that this is an OPT model.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    select_trainable_parameters(model, args.trainable)

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Trainable parameters: {trainable_params:,}")

    dataloader = build_dataloader(tokenizer)
    mozo_args = SimpleNamespace(
        mozo_population_size=args.population_size,
        num_objectives=2,
        zo_eps=args.zo_eps,
        mozo_step_size=args.mozo_step_size,
        eval_batches=args.eval_batches,
    )
    trainer = MinimalMOZOTrainer(model, dataloader, mozo_args)

    for step in range(1, args.max_steps + 1):
        batch = next(iter(dataloader))
        best_objectives = mozo_step(trainer, model, batch)
        print(f"Step {step}: best [loss, l2] = {best_objectives}")


if __name__ == "__main__":
    main()
