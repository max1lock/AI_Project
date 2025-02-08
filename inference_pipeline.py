import os
import cv2
import argparse
from picsellia import Client, ModelVersion
from ultralytics import YOLO


def load_model_from_picsellia(
    api_key: str, workspace_name: str, model_name: str
) -> str:
    """
    Récupère la version du modèle depuis Picsellia et télécharge le fichier best.pt.

    Args:
        api_key (str): Clé API de Picsellia.
        workspace_name (str): Nom de l'organisation sur Picsellia.
        model_name (str): Nom du modèle à récupérer.

    Returns:
        str: Chemin local du fichier best.pt.
    """
    client = Client(api_token=api_key, organization_name=workspace_name)

    model_name = "Groupe_6"
    model = client.get_model(name=model_name)

    model_version = model.get_version(model.latest_version)

    model_path = "./best.pt"
    model_version.download(target_path=model_path)
    print(f"✅ Modèle téléchargé : {model_path}")

    return model_path


def run_inference(model_path: str, mode: str, source: str = None):
    """
    Effectue une inférence avec le modèle YOLO sur une image, une vidéo ou une webcam.

    Args:
        model_path (str): Chemin vers le modèle best.pt.
        mode (str): Mode d'inférence (IMAGE, VIDEO, WEBCAM).
        source (str, optional): Chemin du fichier pour IMAGE ou VIDEO. Non requis pour WEBCAM.
    """
    model = YOLO(model_path)

    if mode == "IMAGE":
        if source is None:
            raise ValueError("❌ Chemin d'image requis pour le mode IMAGE.")
        results = model(source)
        results.show()  # Affiche les résultats
        print("✅ Inférence sur IMAGE terminée.")

    elif mode == "VIDEO":
        if source is None:
            raise ValueError("❌ Chemin de vidéo requis pour le mode VIDEO.")
        results = model(source, stream=True)
        for frame in results:
            frame.show()  # Affiche chaque frame analysée
        print("✅ Inférence sur VIDEO terminée.")

    elif mode == "WEBCAM":
        cap = cv2.VideoCapture(0)
        print(
            "🎥 Inférence en temps réel via WEBCAM... (Appuyez sur 'q' pour quitter)"
        )
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            cv2.imshow("YOLO Inference", results.render()[0])
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Inférence WEBCAM terminée.")

    else:
        raise ValueError(
            "❌ Mode invalide. Choisissez IMAGE, VIDEO ou WEBCAM."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline d'inférence avec YOLO et Picsellia."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["IMAGE", "VIDEO", "WEBCAM"],
        help="Mode d'inférence.",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=False,
        help="Chemin du fichier (requis pour IMAGE et VIDEO).",
    )

    args = parser.parse_args()

    # Connexion et récupération du modèle
    API_KEY = "6a0008897629d989cf385e35ff3c60e45b355584"
    WORKSPACE_NAME = "Picsalex-MLOps"
    MODEL_NAME = "Groupe_6"

    model_path = load_model_from_picsellia(API_KEY, WORKSPACE_NAME, MODEL_NAME)

    # Exécuter l'inférence
    run_inference(model_path, args.mode, args.source)
