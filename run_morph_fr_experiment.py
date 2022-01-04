from face_reco_experiments.morph_reco import run_fr_experiment
from fr.dlib import DlibFr
from datetime import datetime

run_fr_experiment(
    DlibFr(), f"{datetime.now().strftime('%Y%m%d%H%M%S')}_dlib_fr_results.json"
)
