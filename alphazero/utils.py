import os
from tqdm import tqdm
import numpy as np
from huggingface_hub import HfApi, hf_hub_download

import alphazero

DEFAULT_USER_HF = "t0m1ab"
DEFAULT_MODEL_PATH = os.path.join(alphazero.__path__[0], "models/")
DEFAULT_TOKEN_ENV_VAR_NAME = "HF_HUB_TOKEN"


class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    def __getattr__(self, name):
        return self[name]


def fair_max(elements: list | np.ndarray, key = lambda x: x) -> int:
    """
    Returns the index of the maximum element in the collection <elements> by randomly resolving ties.
    """
    max_value = key(max(elements, key=key))
    max_elements = [x for x in elements if key(x) == max_value]
    return max_elements[np.random.choice(len(max_elements))]


def remove_ext(filename: str) -> str:
    """ Remove the extension of a filename. """
    return filename.split(".")[0]


def get_hf_token(token: str = None) -> str:
    """ Get the Hugging Face token stored as an env variable if <token> is None else returns <token>. """
    if token is None and DEFAULT_TOKEN_ENV_VAR_NAME in os.environ:
        token = os.environ[DEFAULT_TOKEN_ENV_VAR_NAME]
    return token


def list_models_from_hf_hub(
        author: str = None, 
        prefix: str = "alphazero", 
        token: str = None
    ) -> list:
    """ List all the models available on the Hugging Face Hub linked to this project. """
    
    author = DEFAULT_USER_HF if author is None else author
    api = HfApi(token=get_hf_token(token)) # will only see public models if no token is provided
    all_models = api.list_models(author=author)
    
    model_ids = [] # filter models and only keep those related to alphazero
    for model in all_models:
        if prefix is None or (prefix is not None and model.id.startswith(f"{author}/{prefix}-")):
            model_ids.append(model.id)

    return model_ids


def download_model_from_hf_hub(
        model_id: str,
        models_dir: str = None, 
        token: str = None,
        verbose: bool = False,
    ) -> str:
    """ 
    Download model matching <model_id> from the Hugging Face Hub and save it in <models_dir>. 
    """

    token = get_hf_token(token)
    model_ids = list_models_from_hf_hub(prefix=None, token=token) # at least all public models
    if model_id not in model_ids:
        raise ValueError(f"Model {model_id} not found on the Hugging Face Hub...")

    models_dir = DEFAULT_MODEL_PATH if models_dir is None else models_dir
    model_name = model_id.split("/")[-1]

    # download the model weights
    _ = hf_hub_download(
        repo_id=model_id,
        repo_type="model",
        filename=f"{model_name}.pt",
        local_dir=os.path.join(models_dir, model_name),
        token=token,
    )

    # download the config file
    _ = hf_hub_download(
        repo_id=model_id,
        repo_type="model",
        filename="config.json",
        local_dir=os.path.join(models_dir, model_name),
        token=token,
    )

    if verbose:
        print(f"Model {model_id} downloaded in: {models_dir}")


def download_all_models_from_hf_hub(
        models_dir: str = None, 
        token: str = None,
        verbose: bool = False,
    ) -> str:
    """ 
    Download all models linked to the project available in the HF Hub and save them in <models_dir>. 
    """

    model_ids = list_models_from_hf_hub() # deafult author is DEFAULT_USER_HF

    if len(model_ids) == 0:
        print(f"No alphazero models found on the Hugging Face Hub... A token (stored in env var {DEFAULT_TOKEN_ENV_VAR_NAME}) may be required to access private models.") if verbose else None
        return

    for model_id in model_ids:
        download_model_from_hf_hub(
            model_id=model_id,
            models_dir=models_dir,
            token=token,
            verbose=verbose,
        )
    print(f"Successfully downloaded {len(model_ids)} models from the Hugging Face Hub!") if verbose else None


def push_model_to_hf_hub(
        model_name: str,
        author: str = None,
        models_dir: str = None, 
        token: str = None,
        private: bool = True,
        additional_files: list[str] = None,
        verbose: bool = False,
    ) -> None:
    """
    Push folder named <model_name> located at <models_dir> to the Hugging Face Hub.
    <additional_files> is a list of files to be pushed to the Hub along with the model.
    """

    models_dir = DEFAULT_MODEL_PATH if models_dir is None else models_dir
    author = DEFAULT_USER_HF if author is None else author
    token = get_hf_token(token)

    # get the path to the model and config files
    path_to_pt_file = os.path.join(models_dir, model_name, f"{model_name}.pt")
    path_to_config_file = os.path.join(models_dir, model_name, "config.json")
    if not os.path.isfile(path_to_pt_file):
        raise ValueError(f"Model file not found: {path_to_pt_file}")
    if not os.path.isfile(path_to_config_file):
        raise ValueError(f"Config file not found: {path_to_config_file}")
    
    # get the path to the additional files
    additional_files = [] if additional_files is None else additional_files
    add_files_dict = {}
    for add_file in additional_files:
        path_to_add_file = os.path.join(models_dir, model_name, add_file)
        if not os.path.isfile(path_to_add_file):
            raise ValueError(f"Additional file not found: {path_to_add_file}")

    # create the model repository on the HFhub
    repo_id = "/".join([author, model_name])
    hfapi = HfApi(token=token)
    hfapi.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
    )

    # push the model weights
    hfapi.upload_file(
        path_or_fileobj=path_to_config_file,
        path_in_repo="config.json",
        repo_id=repo_id,
        repo_type="model",
        commit_message="add config.json",
    )

    # push the config file
    hfapi.upload_file(
        path_or_fileobj=path_to_pt_file,
        path_in_repo=f"{model_name}.pt",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"add {model_name}.pt",
    )

    # push any additional files
    for add_file, path_to_add_file in add_files_dict.items():
        hfapi.upload_file(
            path_or_fileobj=path_to_add_file,
            path_in_repo=add_file,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"add {add_file}",
        )
    
    if verbose:
        print(f"Model '{repo_id}' was sucessfully pushed to the Hugging Face Hub!")


def tests():
    # test fair_max (at least it is a max operator)
    elements = [1, 2, 3, 3, 3, -4, -5]
    assert fair_max(elements) == 3


def main():
    # print(list_models_from_hf_hub())
    # push_model_to_hf_hub(model_name="alphazero-fake", verbose=True)
    download_all_models_from_hf_hub(verbose=True)


if __name__ == "__main__":
    main()