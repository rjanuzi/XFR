import trace
import traceback
from fr.distances_generator import gen_dlib_distances
from util._telegram import send_simple_message
import traceback


if __name__ == "__main__":
    try:
        gen_dlib_distances()
        send_simple_message("Gen DLIB distances matrix done!")
    except:
        send_simple_message("Error generating DLIB distances matrix.")
        print(traceback.format_exc())
