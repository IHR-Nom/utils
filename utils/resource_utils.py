import json
import os

import requests
from requests.adapters import HTTPAdapter

CHUNK_SIZE = 32768
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.dirname(dir_path)


def get_resource(path, create_parent_dir=False):
    """
    Get the abs path from the resource folder
    """
    abs_path = os.path.join(dir_path, 'resources', *path.split('/'))
    if create_parent_dir:
        parent_dir = os.path.dirname(abs_path)
        os.makedirs(parent_dir, exist_ok=True)
    return abs_path


def get_checkpoint_file_path(name):
    checkpoint_file = get_resource(os.path.join('checkpoints', name), create_parent_dir=True)
    if os.path.isfile(checkpoint_file):
        return checkpoint_file
    subject, mode = tuple(name.replace('.ckpt', '').split('_'))
    file_id = pretrained[subject][mode]
    download_file_from_google_drive(file_id, checkpoint_file)
    return checkpoint_file


def download_file_from_google_drive(file_id, destination):
    base_url = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=10))
    response = session.get(base_url, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(base_url, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    with open(destination, "wb") as fb:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                fb.write(chunk)


def get_cache_path(name):
    return get_resource(os.path.join('cache', name), create_parent_dir=True)
