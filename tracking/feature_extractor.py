import onnxruntime
import cv2
import numpy as np
from torchvision import transforms
import torchvision.transforms as T
from PIL import Image
import albumentations as A
from scipy.spatial import distance
from albumentations.pytorch import ToTensorV2
from utils.logger import Logger
logger = Logger()

# _C.MODEL.PIXEL_MEAN = [0.485*255, 0.456*255, 0.406*255]
# # Values to be used for image normalization
# _C.MODEL.PIXEL_STD = [0.229*255, 0.224*255, 0.225*255]

class FeatureExtractor:
    def __init__(self, model_path, device='cuda', gpu_id=0) -> None:
        print("[INFO] ReID ONNX All Providers: ", onnxruntime.get_available_providers())
        print("[INFO] ReID ONNX detected devices: ", onnxruntime.get_device())
        opt_session = onnxruntime.SessionOptions()
        opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.device = device
        self.gpu = gpu_id
        providers = ['CPUExecutionProvider']
        if self.device.casefold() != "cpu":
            # providers.append("CUDAExecutionProvider")
            # providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider']
            logger.info(f"[INFO] ReID Initiating on GPU ID: {self.gpu}")
            providers=[ ('CUDAExecutionProvider',  {
                            'device_id': self.gpu,
                        })
                    , 'CPUExecutionProvider']
        else:
            logger.info("[INFO] ReID Initiating on CPU ")
        logger.info(f"Loading ReID model at: {model_path}")
        self.ort_sess = onnxruntime.InferenceSession(model_path, providers=providers)
        # self.ort_sess = onnxruntime.InferenceSession(model_path)
        self.input_name = self.ort_sess.get_inputs()[0].name
        self.input_shape = self.ort_sess.get_inputs()[0].shape
        self.input_width = self.input_shape[3]
        self.input_height = self.input_shape[2]
        self.tranform = self._tranform()
        self.image_augmentation = A.Compose(
            [A.Resize(self.input_height, self.input_width), A.Normalize(), ToTensorV2()]
        )
        self.run_fake_inference()
        logger.info("[INFO] ReID initialized the first inference")

    def _tranform(self):
        val_transforms = T.Compose([
            T.Resize((self.input_height, self.input_width), interpolation=3),
            T.ToTensor(),
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        return val_transforms
    
    def preprocess_tran_reid(self, opencv_image, image_height, image_width):
        image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB) # convert to RGB
        image = Image.fromarray(image.astype(np.uint8))
        img = self.tranform(image)
        img  = img.unsqueeze(0)
        return img.numpy()
    
    def preprocess_agw(self, opencv_image, image_height, image_width):
        # the model expects RGB inputs
        image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        # Apply pre-processing to image.
        img = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
        # mean = [127.5,127.5,127.5]
        # # Values to be used for image normalization
        # std = [127.5,127.5,127.5]
        # img = (img.astype("float32") - mean)/std
        img = img.astype("float32").transpose(2, 0, 1)[np.newaxis]  # (1, 3, h, w)
        return img
    
    def _preprocessing_img(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # image = img[:, :, ::-1] # convert to RGB
        image = self.image_augmentation(image=np.array(image))["image"]
        image = np.expand_dims(image, axis=0)
        return image
    
    def normalize(self, nparray, order=2, axis=-1):
        """Normalize a N-D numpy array along the specified axis."""
        norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
        return nparray / (norm + np.finfo(np.float32).eps)
    
    def extract(self, image, normalize = True):
        image = self.preprocess_tran_reid(image, self.input_height, self.input_width)
        feat = self.ort_sess.run(None, {self.input_name: image})[0]
        # if normalize:
        #     feat = self.normalize(feat, axis=1)
        return feat
    # find non-increasing
    
    # def extract(self, img):
    #     image = self._preprocessing_img(img)
    #     feat = self.ort_sess.run(None, {self.input_name: image})[0]
    #     feat = self.normalize(feat, axis=1)
    #     return feat
    
    def run_fake_inference(self):
        image = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        self.extract(image)