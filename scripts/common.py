# -*- coding: utf-8 -*-
import os
import glob
import json
import base64


def get_filenames(parent_dir, ext='.jpg', rm_ext=False):
    paths = glob.glob(os.path.join(parent_dir, '*%s' % ext))
    if rm_ext:
        paths = [os.path.splitext(path)[0] for path in paths]

    return sorted([os.path.basename(path) for path in paths])


def read_json(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    return data


def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def base64_encode(lbl):
    return base64.urlsafe_b64encode(lbl.encode('utf-8')).decode('utf-8')


def base64_decode(b64_lbl):
    return base64.urlsafe_b64decode(b64_lbl).decode('utf-8')
