import os
from elasticsearch import Elasticsearch
import json
import time
# from utils.logger import Logger
# logger = Logger()

username = 'elastic'
password = 'elastic'
# password = os.getenv('ELASTIC_PASSWORD') # Value you set in the environment variable


class DBManager(object):
    def __init__(self,camid, feature_dims, create_index=False) -> None:
        self.camid = camid
        self.init_db(camid, feature_dims, create_index=create_index)
        # self.index_settings()

    def init_db(self, camid, feature_dims, create_index=False):
        self.client = Elasticsearch(
        "http://localhost:9200",
        basic_auth=(username, password)
        )
        self.camid_index = f'index-searching-{self.camid}'
        self.feature_dims = feature_dims
        print(self.client.info())
        print("Connected to ElasticSearch successfully!!!")
        if create_index:
            # logger.info("Connected to ElasticSearch successfully!!!"
            self.create_index(self.feature_dims, re_create=True)

    def create_index(self, dims, re_create=False):
        resp = self.client.indices.exists(
            index=self.camid_index,
        )
        if resp:
            print(f"Index {self.camid_index} already existed!!!")
            if re_create:
                self.client.indices.delete(index=self.camid_index,)
                print(f"Deleted index {self.camid_index} successfully!!!")
            else:
                return
        resp = self.client.indices.create(
                index= self.camid_index,
                mappings={
                    "properties": {
                        "feature": {
                            "type": "dense_vector",
                            "dims": dims,
                            "element_type": "float",
                            "index": True,
                            "index_options": {
                                "type": "int8_hnsw"
                            },
                            "similarity": "l2_norm"
                        },
                        "saved_image_path":{
                            "type":"text"
                        },
                        "enter_time":{
                            "type":"date"
                        },
                        "leave_time":{
                            "type":"date"
                        }
                    },
                    "_source": {
                        "excludes": [
                            "feature"
                        ]
                    }
                },
            )
        if resp['acknowledged']:
            print(f"Created index {self.camid_index} successfully!!!")
        else:
            print("Create index failed!!!")

    def get_id_data(self, id):
        resp = self.client.get(index=self.camid_index, id=id)
        if resp['found']:
            return resp['_source']
        return None
    
    def add_new_id_data(self, id, data):
        resp = self.client.index(index=self.camid_index, id=id, document=data, refresh=True)
        return resp
    
    def add_new_id(self, id, feature, start_time, leave_time, saved_image_path):
        data = {
            "feature": feature,
            "saved_image_path": saved_image_path,
            "enter_time": start_time,
            "leave_time": leave_time
        }
        resp = self.client.index(index=self.camid_index, id=id, document=data, refresh=True)
        return resp
    
    def update_id_data(self, id, data):
        resp = self.client.update(index=self.camid_index, id=id, doc=data, refresh=True)
        return resp
    
    def update_old_id(self, id, feature, leave_time, saved_image_path):
        update_body = {
            'feature': feature,
            'leave_time': leave_time,
            'saved_image_path': saved_image_path
        }
        resp = self.client.update(index=self.camid_index, id=id, doc=update_body, refresh=True)
        return resp
    
    def search_feature(self, feature, num_candicates, start_time=None, end_time=None):
        resp = self.client.search(
                index=self.camid_index,
                # fields=[
                #     "saved_image_path",
                #     # "enter_time",
                #     # "leave_time",
                # ],
                # query = {
                #     # "size": 1,  # Retrieve only the most recent document
                #     "sort": [
                #         {
                #             "_version": {
                #                 "order": "desc"  # Sort by timestamp in descending order
                #             }
                #         }
                #     ]
                # },
                knn={
                    "field": "feature",
                    "query_vector": feature,
                    "k": num_candicates,
                    "num_candidates": num_candicates,
                    "filter": {
                        "bool": {
                            "filter": [
                                {
                                    "range": {
                                        "enter_time": {
                                            "gte": start_time,
                                            "lte": end_time
                                        }
                                    }
                                }
                            ]
                        },
                        "bool": {
                            "filter": [
                                {
                                    "range": {
                                        "leave_time": {
                                            "gte": start_time,
                                            "lte": end_time
                                        }
                                    }
                                }
                            ]
                        }
                    }
                },
            )
        return resp['hits']['hits']
    
    def is_existed(self, id):
        exists = self.client.exists(index=self.camid_index, id=id)
        return exists
    
    def refresh(self):
        resp = self.client.indices.refresh(
            index=self.camid_index,
        )
        # print(resp)
        return resp

    def del_gid(self, id):
        self.client.delete(index=self.camid_index, id=id)

    def index_settings(self):
        settings = {
        "settings": {
            "index": {
                "refresh_interval": -1  # e.g., "30s"
                }
            }
        }

        # Update the index settings
        response = self.client.indices.put_settings(index=self.camid_index, body=settings)

    def play(self):
        resp = self.client.indices.get( 
            index="_all",
        )
        print(resp)
def test_db():
   

    with open('./data/elastic_data_sct.json', 'r') as file:
    # with open('./data/sample_data.json', 'r') as file:
        data = json.load(file)
    
    list_cam_db = {}
    for camid in data.keys():
        list_cam_db[camid] = DBManager(camid, 3, create_index=True)

    for camid, cam_data in data.items():
        for id, data in cam_data.items():
            # list_cam_db[camid].add_new_id_data(id, data)
            list_cam_db[camid].add_new_id_data(id, data)
            # list_cam_db[camid].add_new_id(id, data["feature"], data["enter_time"],data["leave_time"],data["saved_image_path"])

            # list_cam_db[camid].refresh()

    # print("Getting...")
    resp = list_cam_db['cam0'].get_id_data(1)
    print(resp)
    resp['leave_time'] = "2025-11-28T12:34:59.789123"
    resp['feature'] = [7999,8, 10]
    # resp = list_cam_db['cam0'].update_id_data(1, resp)
    resp = list_cam_db['cam0'].update_old_id(1, resp['feature'], resp['leave_time'], 'output/1.jpg')
    resp = list_cam_db['cam0'].get_id_data(1)
    print(resp)
    print("Searching feature ...")
    feature = [7999,8, 10]
    for cam, db in list_cam_db.items():
        resp = db.search_feature(feature, 3, "2024-11-28T12:34:56.789123", "2026-11-28T12:34:59.789123")
        resp = db.search_feature(feature, 3)

        print("====> Result in camera: ", cam)
        for i, res in enumerate(resp):
            print(res['_id'], ', score: ', res['_score'], res['_source'])
            # print(f'top {i+1}: ', ', id: ',res['_id'], ', score: ', res['_score'], ', image path: ', res['_source'])

        
if __name__ == "__main__":
    test_db()