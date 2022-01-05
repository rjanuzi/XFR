import sys
from datetime import datetime
from pathlib import Path

from face_reco_experiments.morph_reco import run_fr_experiment
from fr.dlib import DlibFr

if len(sys.argv) == 2:
    backup_file = Path(sys.argv[1])
elif len(sys.argv) == 1:
    backup_file = None
else:
    raise ValueError("Wrong number of arguments")

run_fr_experiment(
    ifr=DlibFr(),
    output_file_path=f"{datetime.now().strftime('%Y%m%d%H%M%S')}_dlib_fr_results.json",
    backup_file=backup_file,
)
