"""Settings shared by every stage.

Machine-specific locations and credentials come from the environment, so nothing personal or
host-specific is committed; the defaults assume this repo sits next to the ml-challenge repo.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ML_CHALLENGE_SRC = Path(os.environ.get('ML_CHALLENGE_SRC', ROOT.parent / 'ml-challenge' / 'src'))
WORK = Path(os.environ.get('LINKAGE_WORK', ROOT / 'work'))
PIPELINE_CONFIG = ROOT / 'configs' / 'pipeline.json'
ENCODER = os.environ.get('LINKAGE_ENCODER', 'intfloat/multilingual-e5-small')

DB = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'root'),
    'password': os.environ.get('MYSQL_PWD', ''),
    'database': os.environ.get('DB_NAME', 'mis_40_50'),
}

# More people than this sharing one exact address are a building, campus or hostel, not a household.
GENERIC_MAX = 6
# Share of pincodes never used for training, so evaluation there measures generalisation.
HOLDOUT_SHARE = 0.2
SEED = 11
# An address part seen this often (a town, district or state) is a locality, not a specific place.
# prep.py uses 40 over S1 and query records together; every address is both here, so 20 is the same cut.
LOCALITY_MIN = 20
# Pairs below this score are left out of the report.
REPORT_MIN_SCORE = 0.3


def path(*parts):
    """A location inside the work directory (all outputs, all personal data, never committed)."""
    return WORK.joinpath(*parts)
