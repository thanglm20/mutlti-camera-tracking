import numpy as np
from scipy.optimize import linear_sum_assignment
import lap #linear assignment problem solver
from utils.config import Config
from tracking.metrics import compute_euclidean_distance

class MCT(object):
    def __init__(self) -> None:
        self.global_id = 0
        self.cfg = Config.get_instance()
        self.threshold = self.cfg.reid.getfloat("threshold", 0.5)
        print("Threshold: ", self.threshold)
        self.max_features_len = self.cfg.reid.getint("max_features_len", 5)

        self.id_map = dict()
        '''
        id: 
            {
                "camid": camid,
                "gid": global_id,
                "features": [feature,...],
                "bbox": bbox,
            }
        '''
        
    def make_new_global_id(self, camid, bbox, feature):
        self.global_id += 1
        return {
            "camid": camid,
            "gid": self.global_id,
            "features": [feature],
            "bbox": bbox,
            "tracks": { camid: [((bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2)]}
        }
    
    def update_track(self, camid, gid, bbox):
        if camid in self.id_map[gid]["tracks"]:
            self.id_map[gid]["tracks"][camid].append(((bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2))
        else:
            self.id_map[gid]["tracks"][camid] = [((bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2)]
        if len(self.id_map[gid]["tracks"][camid]) > self.max_features_len:
            self.id_map[gid]["tracks"][camid].pop(0)

    def cost_matrix(self, features, clusters):
        m = len(features)
        n = len(clusters)

        # Initialize the cost matrix with shape (m, n)
        cost_matrix = np.zeros((m, n))
        
        # Compute the cost (e.g., Euclidean distance) between each feature and each cluster center
        for i in range(m):
            for j in range(n):
                # Assuming features[i] and clusters[j] are vectors (e.g., numpy arrays)
                distances = []
                for f in clusters[j][1]:
                    # distances.append(np.linalg.norm(f - features[i]))
                    distances.append(compute_euclidean_distance(f, features[i]))
                cost_matrix[i, j] = np.mean(distances)
        return cost_matrix   
        
    def linear_assignment(self, cost_matrix):
        try:
            _, x, y = lap.lapjv(cost_matrix, extend_cost = True)
            return np.array([[y[i],i] for i in x if i>=0])
        except ImportError:
            x,y = linear_sum_assignment(cost_matrix)
            return np.array(list(zip(x,y)))

    def update(self, camid, identities, bboxes, features):
        ids = []
        out_bboxes = []
        tracks = []
        if len(self.id_map) == 0:
            # assign new global ids to this camera
            for id, bbox, feature in zip(identities, bboxes, features):
                new_person = self.make_new_global_id(camid, bbox, feature)
                self.id_map[new_person["gid"]] = new_person
            
            for gid, p in self.id_map.items():
                ids.append(gid)
                out_bboxes.append(p["bbox"])
                tracks.append(p["tracks"][camid])
            
        else:
            # get cluster[id: features]
            clusters = []
            for gid , item in self.id_map.items():
                clusters.append((gid, item["features"]))

            distance_matrix = self.cost_matrix(features, clusters)
            matched_indices = self.linear_assignment(distance_matrix)
            matched_ids = []
            for i,j in matched_indices:
                matched_ids.append(identities[i])
                if distance_matrix[i][j] < self.threshold:
                    gid = clusters[j][0]
                    self.id_map[gid]["bbox"] = bboxes[i]
                    self.id_map[gid]["features"].append(features[i])
                    self.update_track(camid, gid, bboxes[i])
                    if(len(self.id_map[gid]["features"]) > self.max_features_len):
                        self.id_map[gid]["features"].pop(0)
                    ids.append(gid)
                    out_bboxes.append(self.id_map[gid]["bbox"])
                    tracks.append(self.id_map[gid]["tracks"][camid])

                else:
                    new_person = self.make_new_global_id(camid, bboxes[i], features[i])
                    self.id_map[new_person["gid"]] = new_person
                    ids.append(new_person["gid"])
                    out_bboxes.append(new_person["bbox"])
                    tracks.append(new_person["tracks"][camid])

            unmatched_indices = []
            for idx, id in enumerate(identities):
                if id not in matched_ids:
                    unmatched_indices.append(idx)
            for i in unmatched_indices:
                new_person = self.make_new_global_id(camid, bboxes[i], features[i])
                self.id_map[new_person["gid"]] = new_person
                ids.append(new_person["gid"])
                out_bboxes.append(new_person["bbox"])
                tracks.append(new_person["tracks"][camid])
        return ids, out_bboxes, tracks