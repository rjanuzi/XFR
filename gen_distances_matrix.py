import traceback

from fr.distances_generator import gen_dlib_distances, gen_hog_distances
from util._telegram import send_simple_message

import logging

logging.basicConfig(
    filename="gen_distances_matrix.log",
    format="%(name)s - %(levelname)s - %(message)s",
)

if __name__ == "__main__":
    try:
        logging.info("Starting DLIB distances calculation")
        gen_dlib_distances()
        logging.info("Starting HOG distances calculation")
        gen_hog_distances()
        send_simple_message("Gen distances matrix done!")
        logging.info("All distances calculation done!")
    except:
        send_simple_message("Error generating distances matrix.")
        logging.error(traceback.format_exc())
