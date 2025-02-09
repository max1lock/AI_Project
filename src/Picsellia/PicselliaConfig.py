from picsellia import Client


def get_client(api_key: str, workspace_name: str) -> Client:
    """
    Crée et retourne un client Picsellia.
    """
    return Client(api_token=api_key, organization_name=workspace_name)


def get_project(client: Client, project_name: str):
    """
    Retourne le projet spécifié.
    """
    return client.get_project(project_name=project_name)


def get_model_object(client: Client, model_name: str):
    """
    Retourne l'objet modèle correspondant.
    """
    return client.get_model(name=model_name)
