import os
from elasticsearch import Elasticsearch
import json
# from utils.logger import Logger
# logger = Logger()

username = 'elastic'
password = 'elastic'
# password = os.getenv('ELASTIC_PASSWORD') # Value you set in the environment variable


class DBManager(object):
    def __init__(self, feature_dims) -> None:
        self.init_db(feature_dims)

    def init_db(self, feature_dims):
        self.client = Elasticsearch(
        "http://localhost:9200",
        basic_auth=(username, password)
        )
        self.mct_index = "mtc-index-test"
        print(self.client.info())
        # logger.info("Connected to ElasticSearch successfully!!!")
        print("Connected to ElasticSearch successfully!!!")
        self.feature_dims = feature_dims
        self.create_index(self.feature_dims, re_create=True)
    def create_index(self, dims, re_create=False):
        resp = self.client.indices.exists(
            index=self.mct_index,
        )
        if resp:
            print(f"Index {self.mct_index} already existed!!!")
            if re_create:
                self.client.indices.delete(index=self.mct_index,)
                print(f"Deleted index {self.mct_index} successfully!!!")
            else:
                return
        resp = self.client.indices.create(
                index= self.mct_index,
                mappings={
                    "properties": {
                        "feature": {
                            "type": "dense_vector",
                            "dims": dims,
                            "element_type": "float",
                            # "index": True,
                            # "index_options": {
                            #     "type": "int8_hnsw"
                            # },
                            "similarity": "l2_norm"
                        },
                    },
                    "_source": {
                        "excludes": [
                        "feature"
                        ]
                    }
                },
            )
        if resp['acknowledged']:
            print(f"Created index {self.mct_index} successfully!!!")
        else:
            print("Create index failed!!!")

    def search_feature(self, feature, num_candicates):
        resp = self.client.search(
                index=self.mct_index,
                knn={
                    "field": "feature",
                    "query_vector": feature,
                    "k": num_candicates,
                    "num_candidates": num_candicates
                },
            )
        return resp['hits']['hits']
    
    def update_gid_feature(self, gid, feature):
        resp = self.client.index(index=self.mct_index, id=gid, document={"feature": feature})
        return resp
    def add_new_data(self, gid, data):
        resp = self.client.index(index=self.mct_index, id=gid, document=data)
        return resp
    def add_new_camera_in_gid(self, gid, camera, data):
        pass

    def update_gid_in_camera(self, gid, camera, data):
        pass
        
    def get_gid_data(self, gid, camera=None):
        resp = self.client.get(index=self.mct_index, id=gid)
        if camera is None:
            return resp["_source"]
        else:
            return resp["_source"][camera]

    def refresh(self):
        resp = self.client.indices.refresh(
            index=self.mct_index,
        )
        print(resp)

    def get_all(self):
        doc = {
            'size' : 10000,
            'query': {
                'match_all' : {}
            }
        }
        res = self.client.search(index=self.mct_index, body=doc,scroll='1m')
        print(res)
        scroll = res['_scroll_id']
        res2 = self.client.scroll(scroll_id = scroll, scroll = '1m')
        res =  res['hits']['hits']

    def del_gid(self, gid):
        self.client.delete(index=self.mct_index, id=gid)
    
    def play(self):
        resp = self.client.indices.get( 
            index="_all",
        )
        print(resp)

def test_db():
    db = DBManager(feature_dims=3)

    with open('./data/elastic_data.json', 'r') as file:
    # with open('./data/sample_data.json', 'r') as file:
        data = json.load(file)
    
    db.play()

    for gid, data in data.items():
        print(data)
        db.update_gid_feature(gid, data["feature"])
    db.refresh()
    print("Searching feature ...")
    resp = db.search_feature([7999,8, 10], 3)
    for i, res in enumerate(resp):
        print(f'top {i+1}: ', ', id: ',res['_id'], ', score: ', res['_score'])

        

if __name__ == "__main__":
    test_db()