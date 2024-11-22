
import time
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
from utils.config import Config
from utils.logger import Logger
import torch
logger = Logger()
from tracking.feature_extractor import FeatureExtractor

def cosine_distance(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return 1 - (dot_product / (norm_a * norm_b))

def compute_euclidean_distance(features, others):
    features = torch.from_numpy(features)
    others = torch.from_numpy(others)
    m, n = features.size(0), others.size(0)
    dist_m = (
            torch.pow(features, 2).sum(dim=1, keepdim=True).expand(m, n)
            + torch.pow(others, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    dist_m.addmm_(1, -2, features, others.t())

    return dist_m.cpu().numpy()[0][0]


def diff(model, p1, p2):
    f1 = model.extract(p1)
    f2 = model.extract(p2)
    euc = compute_euclidean_distance(f1, f2)
    cosin = cosine_distance(f1[0], f2[0])
    return euc, cosin

def display(p1, p2):
    h = 720
    w = 360
    # Ensure images have the same height
    p1 = cv2.resize(p1, (w, h))
    p2 = cv2.resize(p2, (w, h))
    # Concatenate images horizontally
    horizontal_img = cv2.hconcat([p1, p2])
    cv2.imshow(f"Output",horizontal_img)
    cv2.waitKey(1) 
    time.sleep(0.005)


def test_all():
    TEST_DIR = "./reid_data"
    TEST_LOOP = 100
    people= ['p1', 'p2', 'p3', 'p4', 'p5']

    cfg = Config.get_instance()
    reid_gpu = cfg.reid.getboolean("gpu", True)
    reid_model = cfg.reid.getstr("reid_model", "./models/reid.onnx")
    logger.info(f"Loading ReID model at: {reid_model}")
    device = 'cuda' if reid_gpu else 'cpu'
    extractor = FeatureExtractor(reid_model, device, 0)


    list_people_images = {}
    for p in people:
        list_file = os.listdir(os.path.join(TEST_DIR, p))
        list_people_images[p] = [os.path.join(TEST_DIR, p, f) for f in list_file]

    logger.info("Calculate ReID for a person")
    p = 0
    euc_in_one = []
    cosine_in_one = []

    for i in range(0, TEST_LOOP):
        rang = len(list_people_images[people[p]])
        img1 = cv2.imread(list_people_images[people[p]][random.randrange(0, rang)])
        img2 = cv2.imread(list_people_images[people[p]][random.randrange(0, rang)])
        euclid, cosine = diff(extractor, img1, img2)
        euc_in_one.append(round(euclid, 1))
        cosine_in_one.append(round(cosine, 1))
        print(euclid, ' - ', cosine)
        display(img1, img2)

    p1 = 0
    p2 = 4
    logger.info(f"Calculate ReID for person {p1} vs {p2}")
    euc_in_two = []
    cosine_in_two = []

    for i in range(0, TEST_LOOP):
        rang1 = len(list_people_images[people[p1]])
        rang2 = len(list_people_images[people[p2]])
        img1 = cv2.imread(list_people_images[people[p1]][random.randrange(0, rang1)])
        img2 = cv2.imread(list_people_images[people[p2]][random.randrange(0, rang2)])
        euclid, cosine = diff(extractor, img1, img2)
        euc_in_two.append(round(euclid, 1))
        cosine_in_two.append(round(cosine, 1))
        print(euclid, ' - ', cosine)
        display(img1, img2)

    # Create a figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 grid of subplots

    # Plot each histogram
    axes[0, 0].hist(euc_in_one, bins=20, color='blue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title("Euclidean Distance distribution on the same person")

    axes[0, 1].hist(cosine_in_one, bins=20, color='green', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title("Cosine Distance distribution on the same person")

    axes[1, 0].hist(euc_in_two, bins=20, color='red', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title("Euclidean Distance distribution on two people")

    axes[1, 1].hist(cosine_in_two, bins=20, color='purple', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title("Cosine Distance distribution on two people")

    # Adjust layout
    plt.tight_layout()
    formatted_time =  time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"./results/histogram/distance_{formatted_time}.png", dpi=300, bbox_inches='tight')
    # Show the plots
    plt.show()

def test_one():
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
    # test_one()
    test_all()
    logger.info("Ended App!!!")