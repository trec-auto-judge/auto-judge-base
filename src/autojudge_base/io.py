from pathlib import Path
from glob import glob
import json
from collections import defaultdict

from typing import List
from .report import load_report, Report
                

def load_runs_failsave(path: Path)->List[Report]:
    ret = []
    path = path.absolute()
    globs = sorted(glob(f"{path}/*") + glob(f"{path}/*/*") + glob(f"{path}/*/*/*"))
    print(f"globs: {globs}")

    for f in globs:
        print(f"Loading report from {f}")
        ret.extend(load_report(Path(f)))
    return ret


def irds_from_dir(directory):
    from tira.ir_datasets_util import load_ir_dataset_from_local_file
    from ir_datasets import registry

    ds = load_ir_dataset_from_local_file(Path(directory), str(directory))
    if str(directory) not in registry:
        registry.register(str(directory), ds)
    return ds


def load_hf_dataset_config_or_none(c, required_fields):
    from yaml import safe_load

    if c.is_file():
        txt = c.read_text()
        for cfg in txt.split("---"):
            try:
                ret = safe_load(cfg)
                if all(f in ret for f in required_fields):
                    return ret
            except:
                pass
