import numpy as np
import json
from scipy.optimize import linear_sum_assignment
import lap #linear assignment problem solver
from utils.config import Config
from tracking.metrics import compute_euclidean_distance
from tracking.frame_manager import FrameData
import math
from tracking.utils import save_person_image
class MCT(object):
    def __init__(self) -> None:
        self.global_id = 0
        self.cfg = Config.get_instance()
        self.threshold = self.cfg.reid.getfloat("threshold", 0.5)
        self.max_features_len = self.cfg.reid.getint("max_features_len", 5)
        self.gid_cluster = dict()
        '''
        gid: 
            camid,
            {
                "local_id": id
                "features": [feature,...],
                "bbox": [bbox]
                "cropped_images": [image]
                "tracks":[]
                "times": [time]
            }
        '''
        
    def make_new_global_id(self, camid, id, bbox,cropped_image, feature,  time):
        self.global_id += 1
        gid = self.global_id
        self.gid_cluster[gid] = {}
        self.gid_cluster[gid][camid] = \
        {
            "local_id": id,
            "bbox": bbox,
            "features": [feature],
            "cropped_images": [cropped_image],
            "times": [time]
        }
        return gid
    
    def update_gid_re_enter_old_cam(self, gid, camid, id, bbox, cropped_image, feature, time):
        self.gid_cluster[gid][camid]["local_id"] = id
        self.gid_cluster[gid][camid]["bbox"] = bbox
        self.gid_cluster[gid][camid]["cropped_images"].append(cropped_image)
        self.gid_cluster[gid][camid]["features"].append(feature)
        self.gid_cluster[gid][camid]["times"].append(time)
        if(len(self.gid_cluster[gid][camid]["features"]) > self.max_features_len):
            self.gid_cluster[gid][camid]["features"].pop(0)
            self.gid_cluster[gid][camid]["cropped_images"].pop(0)
            self.gid_cluster[gid][camid]["times"].pop(0)



    def add_gid_re_enter_new_cam(self, gid, camid, id, bbox, cropped_image, feature, time):
        self.gid_cluster[gid][camid] = \
        {
            "local_id": id,
            "bbox": bbox,
            "features": [feature],
            "cropped_images": [cropped_image],
            "times": [time]
        }
        
    def get_unmatched_cluster(self, matched_gids):
        list_gids_cam = []
        for gid, list_cam in self.gid_cluster.items():
            if gid not in matched_gids: 
                for cam, item in list_cam.items():
                    list_gids_cam.append((gid, cam, item["features"]))
        return list_gids_cam

    def cost_matrix(self, unmatched_features, list_unmatched_gids_cam):
        m = len(unmatched_features)
        n = len(list_unmatched_gids_cam)

        # Initialize the cost matrix with shape (m, n)
        cost_matrix = np.zeros((m, n))
        
        # Compute the cost (e.g., Euclidean distance) between each feature and each cluster center
        for i in range(m):
            for j in range(n):
                distances = []
                for f in list_unmatched_gids_cam[j][2]: #(gid, camid, features)
                    distances.append(compute_euclidean_distance(f, unmatched_features[i]))
                cost_matrix[i, j] = np.mean(distances)
        return cost_matrix 
      
    def calc_swapped_cost_matrix(self, swapped_features, swapped_gids, camid):
        m = len(swapped_features)
        n = len(swapped_gids)
        cost_matrix = np.zeros((n, n))
        for i in range(m):
            for j in range(n):
                distances = []
                gid = swapped_gids[j]
                for f in self.gid_cluster[gid][camid]["features"]:
                    distances.append(compute_euclidean_distance(f, swapped_features[i]))
                cost_matrix[i, j] = np.mean(distances)
        return cost_matrix
       
    def get_distance_with_old_id(self, camid, id, feature):
        # find cam and id
        for gid, list_cam in self.gid_cluster.items():
            if camid in list_cam and id == list_cam[camid]["local_id"]:
                distances = []
                for f in list_cam[camid]["features"]:
                    # distances.append(np.linalg.norm(f - features[i]))
                    distances.append(compute_euclidean_distance(feature, f))
                return gid, np.mean(distances)
        return -1, float('inf')
    
    def linear_assignment(self, cost_matrix):
        try:
            _, x, y = lap.lapjv(cost_matrix, extend_cost = True)
            return np.array([[y[i],i] for i in x if i>=0])
        except ImportError:
            x,y = linear_sum_assignment(cost_matrix)
            return np.array(list(zip(x,y)))

    def update(self, camid, frame_data:FrameData):

        identities = frame_data.identities
        bboxes = frame_data.boxes
        cropped_people = frame_data.cropped_people
        features = frame_data.features
        frame_time = frame_data.time
        gids = []
        matched_gids  = []
        if len(self.gid_cluster.keys()) == 0 and len(identities):
            # assign new global ids to this camera
            for id, bbox, cropped_person, feature in zip(identities, bboxes, cropped_people, features):
                gid = self.make_new_global_id(camid, id, bbox, cropped_person, feature, frame_time)
            # return new
            for gid, p in self.gid_cluster.items():
                matched_gids.append(gid)
   
        else:
            matched_old_gids = []
            
            swapped_old_ids = []
            swapped_features = []
            swapped_gids = []
            swapped_cropped_people = []

            unmatched_ids = []
            unmatched_features = []
            unmatched_bboxes = []
            unmatched_cropped_people =[]

            # step 1: find old camid and id  matching 
            for id, bbox, cropped_person, feature in zip(identities, bboxes, cropped_people, features):
                gid, dist = self.get_distance_with_old_id(camid, id, feature)
                # match old id and threshold is valid => not swap id
                if gid != -1 and dist <= self.threshold:
                    matched_old_gids.append(gid)
                    # update
                    self.update_gid_re_enter_old_cam(gid,camid,id,bbox,cropped_person,feature,frame_time)
                # found, but ids swapped
                elif gid != -1 and dist > self.threshold:
                    swapped_old_ids.append(id)
                    swapped_features.append(feature)
                    swapped_gids.append(gid)
                    swapped_cropped_people.append(cropped_person)
                # not found => new id
                else:
                    unmatched_ids.append(id)
                    unmatched_bboxes.append(bbox)
                    unmatched_features.append(feature)
                    unmatched_cropped_people.append(cropped_person)
            # step 2: re-assign swap id
            if len(swapped_old_ids) > 0:
                dist_matrix = self.calc_swapped_cost_matrix(swapped_features, swapped_gids, camid)
                re_assign_list = self.linear_assignment(dist_matrix)
                for i,j in re_assign_list:
                    id= swapped_old_ids[i]
                    feature = swapped_features[i]
                    cropped = swapped_cropped_people[i]
                    gid = swapped_gids[j]
                    threshold = dist_matrix[i][j]
                    self.update_gid_re_enter_old_cam(gid, camid, id, bbox, cropped, feature, frame_time)
                    matched_old_gids.append(gid)
            gids = matched_old_gids
            # step 3: match new ids with cluster, in others camera and gid
            # process re-enter id and id appears in many cameras simultaneously
            # because of excluding matched old camid, id
            list_unmatched_gids_cam = self.get_unmatched_cluster(matched_old_gids)
            if len(list_unmatched_gids_cam) > 0 :
                dist_matrix = self.cost_matrix(unmatched_features, list_unmatched_gids_cam)
                if(len(dist_matrix)):
                    matched_gid_cam = self.linear_assignment(dist_matrix)
                    for i,j in matched_gid_cam:
                        id = unmatched_ids[i]
                        bbox = unmatched_bboxes[i]
                        feature = unmatched_features[i]
                        cropped = unmatched_cropped_people[i]
                        matched_gid = list_unmatched_gids_cam[j][0]
                        matched_camid = list_unmatched_gids_cam[j][1]
                        threshold = dist_matrix[i][j]
                        # match
                        if threshold <= self.threshold:
                            # re-enter in the same camera
                            if camid == matched_camid:
                                self.update_gid_re_enter_old_cam(matched_gid, matched_camid, id, bbox, cropped, feature, frame_time)
                            # re-enter in the different camera
                            else:
                                self.add_gid_re_enter_new_cam(matched_gid, camid, id, bbox, cropped,feature, frame_time )
                        # unmatch =>add new gid
                        else:
                            gid = self.make_new_global_id(camid, id, bbox,cropped_person, feature,frame_time)
                            gids.append(gid)
            else:
                for id, bbox, feature in zip(unmatched_ids, unmatched_bboxes, unmatched_features):
                    gid = self.make_new_global_id(camid, id, bbox,cropped_person, feature,frame_time)
                    gids.append(gid)
            # self.save_images("./outputs")
        return gids
    
    def search(self, feature):
        if self.gid_cluster is None:
            return None, None
        distances = []
        for gid, list_cam in self.gid_cluster.items():
            distance = []
            for camid, data in list_cam.items():
                for f in data["features"]:
                    distance.append(compute_euclidean_distance(f, feature))
            distances.append((gid, np.mean(distance)))
        if len(distances) == 0:
            return None, None
        sorted_distacne = sorted(distances, key=lambda x: x[1])
        if(sorted_distacne[0][1] > self.threshold):
            return None, None
        result = {}
        top1_gid = sorted_distacne[0][0]
        result['gid'] = top1_gid
        result['cameras'] = []
        result['last_times'] = []
        cropped_images = []
        for camid, data in self.gid_cluster[top1_gid].items():
            result['cameras'].append(camid)
            result['last_times'].append(data['times'][-1])
            cropped_images.append(data['cropped_images'][-1])
            print(data['cropped_images'][-1].shape)
        return json.dumps(result), cropped_images
    
    def get_gid_clusters(self, gids, camid):
        boxes = []
        tracks = []
        for gid in gids:
            boxes.append(self.gid_cluster[gid][camid]["bbox"])
        return boxes
    

    def save_images(self, output_dir):
        for gid, list_cam in self.gid_cluster.items():
            distance = []
            for camid, data in list_cam.items():
                for i, image in enumerate(data["cropped_images"]):
                    save_person_image(output_dir, camid, i, gid, image)
