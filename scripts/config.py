# -*- coding: utf-8 -*-
from common import read_json


def load_config(config_path):
    configs = list()
    path = config_path

    while True:
        config = read_json(path)
        assert config is not None, '"%s" is not found.' % path

        configs.append(config)

        if 'inherit' in config:
            path = config.get('inherit')
        else:
            break

    configs = configs[::-1]
    config = configs[0]
    for override in configs[1:]:
        merge_config(config, override)

    return config


def merge_config(conf1, conf2):
    def merge_dict(d1, d2):
        for key in d1.keys():
            if key in d2:
                if isinstance(d2[key], list):
                    d1[key] = merge_list(d1[key], d2[key])
                elif isinstance(d2[key], dict):
                    merge_dict(d1[key], d2[key])
                else:
                    d1[key] = d2[key]

    def merge_list(l1, l2):
        for i in range(min(len(l1), len(l2))):
            if isinstance(l2[i], dict):
                merge_dict(l1[i], l2[i])
            elif isinstance(l2[i], list):
                merge_list(l1[i], l2[i])
            else:
                l1[i] = l2[i]
        if len(l1) < len(l2):
            l1.extend(l2[len(l1):])
        else:
            l1 = l1[:len(l2)]
        return l1

    merge_dict(conf1, conf2)
