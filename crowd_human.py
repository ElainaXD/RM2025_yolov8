"""
Based on https://raw.githubusercontent.com/jkjung-avt/yolov4_crowdhuman/master/data/gen_txts.py
Inputs:
    * nothing
    * or folder with CrowdHuman1.zip, CrowdHuman2.zip, CrowdHuman3.zip, CrowdHuman_val.zip, annotation_train.odgt, annotation_val.odgt
python crowdhuman_to_yolo.py --dataset_path foo/bar/
Outputs:
    * same folder with :
        - labels/train 
        - labels/val
        - images/train
        - images/val
"""


import json
from pathlib import Path
from argparse import ArgumentParser
import requests
import os
import zipfile
import numpy as np
import cv2
import shutil

def make_dir_ignore(path):
    try:
        os.makedirs(path)
    except:
        print("")

def download_file_from_google_drive(id, destination):
    #https://stackoverflow.com/a/39225039/7036639
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    save_response_content(response, destination) 


def download_crowd_dataset(output_dir, skip_download=True):
    output_dir = str(output_dir)
    crowd_url_dict = {"CrowdHuman1.zip": "134QOvaatwKdy0iIeNqA_p-xkAhkV4F8Y",
                      "CrowdHuman2.zip": "17evzPh7gc1JBNvnW1ENXLy5Kr4Q_Nnla",
                      "CrowdHuman3.zip": "1tdp0UCgxrqy1B6p8LkR-Iy0aIJ8l4fJW",
                      "CrowdHuman_val.zip": "18jFI789CoHTppQ7vmRSFEdnGaSQZ4YzO",
                      "annotation_train.odgt": "1UUTea5mYqvlUObsC1Z8CFldHJAtLtMX3",
                      "annotation_val.odgt": "10WIRwu8ju8GRLuCkZ_vT6hnNxs5ptwoL"
                      }
    for crowd_file, crowd_file_id  in crowd_url_dict.items():
        crowd_file_path = os.path.join(output_dir, crowd_file)
        if not skip_download and not os.path.isfile(crowd_file_path):
            print("File not found, trying to download it...")
            download_file_from_google_drive(crowd_file_id, crowd_file_path)
        if ".zip" in crowd_file:
            folder_unzipped = crowd_file_path.replace(".zip",'')
            with zipfile.ZipFile(crowd_file_path,"r") as zip_ref:
                zip_ref.extractall(folder_unzipped)
    # Merge image folder
    dest = os.path.join(output_dir, "images", "train")
    make_dir_ignore(dest)
    for train_folder in ["CrowdHuman1", "CrowdHuman2", "CrowdHuman3"]:
        src = os.path.join(output_dir, train_folder, "Images")
        files = os.listdir(src)
        for f in files:
            shutil.move(os.path.join(src,f), dest)
        shutil.rmtree(os.path.join(output_dir, train_folder))
    val_folder = os.path.join(output_dir, "CrowdHuman_val")
    src = os.path.join(val_folder, "Images")
    dest = os.path.join(output_dir, "images", "val")
    make_dir_ignore(dest)
    files = os.listdir(src)
    for f in files:
        shutil.move(os.path.join(src,f), dest)
    shutil.rmtree(val_folder)

def image_shape(ID, image_dir):
    assert image_dir is not None
    jpg_path = image_dir / ('%s.jpg' % ID)
    #print(jpg_path)
    img = cv2.imread(jpg_path.as_posix())
    return img.shape


def txt_line(cls, bbox, img_w, img_h):
    """Generate 1 line in the txt file."""
    x, y, w, h = bbox
    x = max(int(x), 0)
    y = max(int(y), 0)
    w = min(int(w), img_w - x)
    h = min(int(h), img_h - y)

    cx = (x + w / 2.) / img_w
    cy = (y + h / 2.) / img_h
    nw = float(w) / img_w
    nh = float(h) / img_h
    return '%d %.6f %.6f %.6f %.6f\n' % (cls, cx, cy, nw, nh)

def process(set_='val', annotation_filename='annotation_val.odgt',
            output_dir=None):
    """Process either 'train' or 'test' set."""
    assert output_dir is not None
    output_dir.mkdir(exist_ok=True)
    jpgs = []
    make_dir_ignore(output_dir / "labels" / set_)
    
    with open(annotation_filename, 'r') as fanno:
        for raw_anno in fanno.readlines():
            anno = json.loads(raw_anno)
            ID = anno['ID']  # e.g. '273271,c9db000d5146c15'
            #print('Processing ID: %s' % ID)
            img_h, img_w, img_c = image_shape(ID, output_dir / Path("images") / Path(set_))
            assert img_c == 3  # should be a BGR image
            txt_path = output_dir / "labels" / set_ / ('%s.txt' % ID)
            # write a txt for each image
            with open(txt_path.as_posix(), 'w') as ftxt:
                for obj in anno['gtboxes']:
                    if obj['tag'] == 'mask':
                        continue  # ignore non-human
                    assert obj['tag'] == 'person'
                    if 'hbox' in obj.keys():  # head
                        line = txt_line(1, obj['hbox'], img_w, img_h)
                        if line:
                            ftxt.write(line)
                    if 'fbox' in obj.keys():  # full body
                        line = txt_line(0, obj['fbox'], img_w, img_h)
                        if line:
                            ftxt.write(line)
            jpgs.append('%s/%s.jpg' % (output_dir / Path("images") / Path(set_), ID))
    # write the 'data/crowdhuman/train.txt' or 'data/crowdhuman/test.txt'
    set_path = output_dir / ('%s.txt' % set_)
    with open(set_path.as_posix(), 'w') as fset:
        for jpg in jpgs:
            fset.write('%s\n' % jpg)


def rm_txts(output_dir):
    """Remove txt files in output_dir."""
    for txt in output_dir.glob('*.txt'):
        if txt.is_file():
            txt.unlink()


def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset_path', default=r"/app/yolov8pose/CrowdHuman", type=str, help='Dataset name')
    parser.add_argument('--download', action='store_true', help='Download the dataset from GoogleDrive (may fail)')

    args = parser.parse_args()
    print(args)
    output_dir = Path(args.dataset_path)
    if not output_dir.is_dir():
        os.mkdir(output_dir)
    download_crowd_dataset(output_dir, skip_download=(not args.download))

    rm_txts(output_dir)
    process('val', output_dir / Path('annotation_val.odgt'), output_dir)
    process('train', output_dir / Path('annotation_train.odgt'), output_dir)

if __name__ == '__main__':
    main()