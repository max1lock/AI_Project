import argparse
from src.Pipelines.Train.train_pipeline import train_model
from src.Pipelines.Inference.inference_pipeline import run_inference
from src.Pipelines.Inference.inference_pipeline import (
    load_model_from_picsellia,
)


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

        if (
            inference_mode == "image"
        ):  # Si le mode est image, demandez un chemin d'image
            source_path = input("Entrez le chemin de l'image : ").strip()
        elif (
            inference_mode == "video"
        ):  # Si le mode est vidéo, demandez un chemin de vidéo
            source_path = input("Entrez le chemin de la vidéo : ").strip()

        # Charger le modèle depuis Picsellia
        model_path = load_model_from_picsellia(
            api_key="6a0008897629d989cf385e35ff3c60e45b355584",
            workspace_name="Picsalex-MLOps",
            model_name="Groupe_6",
        )

        # Appeler la fonction run_inference avec le mode choisi
        run_inference(
            model_path,
            inference_mode,
            source_path if inference_mode in ["image", "video"] else None,
        )
    else:
        print("Mode invalide. Veuillez choisir 'train' ou 'inference'.")


if __name__ == "__main__":
    main()
