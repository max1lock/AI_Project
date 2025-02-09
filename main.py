import argparse
from src.Pipelines.Train.train_pipeline import train_model
from src.Pipelines.Inference.inference_pipeline import (
    run_inference,
    load_model_from_picsellia,
)
from config import settings


def main():
    """
    Main function to execute the full pipeline of downloading dataset,
    structuring data, training a YOLO model, and evaluating the results.
    """
    parser = argparse.ArgumentParser(
        description="Choisissez une pipeline : entraînement ou inférence."
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--train",
        action="store_true",
        help="Lancer la pipeline d'entraînement.",
    )
    mode_group.add_argument(
        "--inference",
        action="store_true",
        help="Lancer la pipeline d'inférence.",
    )

    inf_mode_group = parser.add_mutually_exclusive_group()
    inf_mode_group.add_argument(
        "--webcam",
        action="store_true",
        help="Utiliser la webcam pour l'inférence.",
    )
    inf_mode_group.add_argument(
        "--image",
        action="store_true",
        help="Utiliser une image pour l'inférence.",
    )
    inf_mode_group.add_argument(
        "--video",
        action="store_true",
        help="Utiliser une vidéo pour l'inférence.",
    )

    parser.add_argument(
        "--source",
        type=str,
        help="Chemin du fichier source pour l'inférence (obligatoire pour image ou vidéo).",
    )

    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.inference:
        if args.image:
            mode = "image"
        elif args.video:
            mode = "video"
        else:
            mode = "webcam"

        if mode in ["image", "video"] and not args.source:
            parser.error(
                f"Pour le mode '{mode}', vous devez fournir --source <chemin>."
            )

        model_path = load_model_from_picsellia(
            api_key=settings.api_key,
            workspace_name=settings.workspace_name,
            model_name=settings.groupe_name,
        )

        run_inference(model_path, mode, args.source)


if __name__ == "__main__":
    main()
