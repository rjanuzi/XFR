import traceback

from fr.distances_generator import gen_dlib_distances, gen_hog_distances
from util._telegram import send_simple_message

if __name__ == "__main__":
    try:
        # gen_dlib_distances()
        gen_hog_distances()
        send_simple_message("Gen distances matrix done!")
    except:
        send_simple_message("Error generating distances matrix.")
        print(traceback.format_exc())
