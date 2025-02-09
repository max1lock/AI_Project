import argparse
from src.Pipelines.Train.train_pipeline import train_model
from src.Pipelines.Inference.inference_pipeline import (
    run_inference,
    load_model_from_picsellia,
)
from config import settings


def main():
    parser = argparse.ArgumentParser(
        description="Choisissez une pipeline : entraînement ou inférence."
    )
    parser.add_argument(
        "mode",
        choices=["train", "inference"],
        help="Mode à exécuter : train ou inference",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "inference":
        inference_mode = (
            input("Choisissez un mode d'inférence (webcam, image, video) : ")
            .strip()
            .lower()
        )
        source_path = None
        if inference_mode in ["image", "video"]:
            source_path = input(
                "Entrez le chemin du fichier source : "
            ).strip()

        model_path = load_model_from_picsellia(
            api_key=settings.api_key,
            workspace_name=settings.workspace_name,
            model_name="Groupe_6",
        )

        run_inference(model_path, inference_mode, source_path)
    else:
        print("Mode invalide. Veuillez choisir 'train' ou 'inference'.")


if __name__ == "__main__":
    main()
