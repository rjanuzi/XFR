import traceback

from experiments.dlib_resnet_ga_approximation import run_experiment
from util._telegram import send_simple_message

try:
    run_experiment()
except:
    print(traceback.format_exc())
    send_simple_message("Error in dlib_resnet_ga_approximation.py")
else:
    send_simple_message("Finished dlib_resnet_ga_approximation.py")
