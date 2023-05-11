import sys
import traceback

from experiments.dlib_resnet_ga_approximation import run_experiment, run_experiment_v2
from util._telegram import send_simple_message
import logging

logging.basicConfig(
    format="%(asctime)s %(message)s",
    filename="run_ga_experiment.log",
    level=logging.INFO,
)

if __name__ == "__main__":
    params = None

    try:
        if sys.argv[1] == "default":
            params = [
                {
                    "cxpb": 0.4,
                    "mutpb": 0.3,
                    "indpb": 0.15,
                    "pop_size": 50,
                    "max_generations": 5,
                    "error_fun": "rank_error",
                }
            ]
    except:
        pass

    logging.info(f"Running experiment with params: {params}")

    try:
        run_experiment_v2(params_comb=params, verbose=False)
    except:
        print(traceback.format_exc())
        logging.error(traceback.format_exc())
        send_simple_message("Error in dlib_resnet_ga_approximation.py")
    else:
        logging.info("Finished dlib_resnet_ga_approximation.py")
        send_simple_message("Finished dlib_resnet_ga_approximation.py")
