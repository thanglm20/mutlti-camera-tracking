
import time
import numpy as np
import cv2
from utils.config import Config
from utils.logger import Logger
logger = Logger()
from tracking.feature_extractor import FeatureExtractor

def cosine_distance(a, b):
    """Calculates the cosine distance between two vectors.

    Args:
        a: The first vector.
        b: The second vector.

    Returns:
        The cosine distance between the two vectors.
    """


    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return 1 - (dot_product / (norm_a * norm_b))


def test_reid():
    cfg = Config.get_instance()
    reid_gpu = cfg.reid.getboolean("gpu", True)
    reid_model = cfg.reid.getstr("reid_model", "./models/reid.onnx")
    logger.info(f"Loading ReID model at: {reid_model}")
    device = 'cuda' if reid_gpu else 'cpu'
    extractor = FeatureExtractor(reid_model, device, 0)

    p1 = cv2.imread("D:/projects/ReID_Survey/mct/outputs/cam0/p156/f454.jpg")
    p2 = cv2.imread("D:/projects/ReID_Survey/mct/outputs/cam0/p156/f453.jpg")

    f1 = extractor.extract(p1)
    f2 = extractor.extract(p2)

    dis = np.linalg.norm(f1 - f2)
    # dis = cosine_distance(f1[0], f2[0])

    logger.info(f'Distance: {dis}')

    h = 720
    w = 360
    # Ensure images have the same height
    p1 = cv2.resize(p1, (w, h))
    p2 = cv2.resize(p2, (w, h))

    # Concatenate images horizontally
    horizontal_img = cv2.hconcat([p1, p2])
    while True:
        cv2.imshow(f"Output",horizontal_img)
        if cv2.waitKey(25) == 'q':
            break

if __name__ == "__main__":
    test_reid()
    logger.info("Ended App!!!")