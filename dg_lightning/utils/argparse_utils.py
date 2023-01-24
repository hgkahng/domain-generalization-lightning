
import os
import yaml
import argparse
import datetime


def override_defaults_given_yaml_config_file(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Helper function used to override defaults of parser given yaml config file."""
    args, _ = parser.parse_known_args()
    if args.config is not None:
        assert isinstance(args.config, list)
        for yaml_file in args.config:
            assert yaml_file.endswith('.yaml')
            with open(yaml_file, 'r') as f:
                cfg = yaml.safe_load(f)                                  # load defaults
                cfg = {k: v for k, v in cfg.items() if k in vars(args)}  # only those in argparse
                parser.set_defaults(**cfg)
    
    return parser


def create_datetime_hash(fmt: str = "%Y-%m-%d_%H:%M:%S") -> str:
    return datetime.datetime.now().strftime(fmt)
