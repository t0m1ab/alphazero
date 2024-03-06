import argparse

from alphazero.utils import download_all_models_from_hf_hub


def main():
    """
    Download all alphazero models from the HF Hub.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dir",
        dest="models_dir",
        type=str,
        default=None,
        help="Specify the path to the local directory where the models will be saved.",
    )
    parser.add_argument(
        "-t",
        "--token",
        dest="token",
        type=str,
        default=None,
        help="Token to use to download the models from the HF Hub. If None, the token is set to the value of the env var DEFAULT_TOKEN_ENV_VAR_NAME if it exists else no token is used.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        dest="quiet",
        action="store_true",
        default=False,
        help="If True then verbose is set to False during download.",
    )
    args = parser.parse_args()

    verbose = not args.quiet

    download_all_models_from_hf_hub(
        models_dir=args.models_dir,
        token=args.token,
        verbose=verbose
    )


if __name__ == "__main__":
    main()
    # Example: python download.py -d /path/to/models -t my_token -q
