from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from demo.predictor import VisualizationDemo
import os
from shutil import move
import cv2
from tqdm import tqdm

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import argparse

CONFIG_FILE = 'configs/ocr/icdar2015_101_FPN.yaml'
OPTS = []
CONFIDENCE_THRESHOLD = 0.5

def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(CONFIG_FILE)
    cfg.merge_from_list(OPTS)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = CONFIDENCE_THRESHOLD
    cfg.freeze()
    return cfg


# img_path = '../london/image'
# new_img_path = '../london/new_image'
# save_path = '../gen_london/image'

def get_parser():
    parser = argparse.ArgumentParser(description="Get image with no text from dataset")
    parser.add_argument("--img-path", type=str, help="path to image dataset")
    parser.add_argument("--new-img-path", type=str, help="path to save image with text")
    parser.add_argument("--save-path", type=str, help="path to save image with no text")
    
    return parser


def main():
    cfg = setup_cfg()
    demo = VisualizationDemo(cfg)
    
    args = get_parser().parse_args()
    img_path = args.img_path
    new_img_path = args.new_img_path
    save_path = args.save_path
    
    for index, path in enumerate(tqdm(os.listdir(img_path))):
        if path.endswith('.jpg'):
            fullpath = os.path.join(img_path, path)
            img = read_image(fullpath, format='BGR')
            # remove the text 'google' from image
            img = img[:620, :, :]
            pred, out, poly = demo.run_on_image(img)
            if poly==[]:
                cv2.imwrite(os.path.join(save_path, path), img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                os.remove(fullpath)

            else:
                cv2.imwrite(os.path.join(new_img_path, path), img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                os.remove(fullpath)
                
                
if __name__=='__main__':
    main()