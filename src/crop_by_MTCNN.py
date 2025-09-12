
import PIL
from mtcnn import MTCNN
import cv2
import numpy as np
import os
import argparse


detector = MTCNN()

def box2square_original(bbox):
    """Convert bbox to square
    Parameters:
    ----------
    bbox: numpy array , shape n x 5
        input bbox
    Returns:
    -------
    square bbox
    """
    square_bbox = bbox.copy()

    h = bbox[3]
    w = bbox[2]
    max_side = np.maximum(h,w)
    square_bbox[0] = bbox[0] + w*0.5 - max_side*0.5
    square_bbox[1] = bbox[1] + h*0.5 - max_side*0.5
    square_bbox[2] = max_side
    square_bbox[3] = max_side
    return square_bbox

def box2square(box):
    if box[2]==box[3]:
        return box
    else:
        center_x=box[0]+0.5*box[2]
        center_y=box[1]+0.5*box[3]
        if box[2]>box[3]:
            new_w = int(box[2])
            new_h = int(box[2])
            new_x = int(box[0])
            new_y = int(center_y - 0.5*new_h)
        elif box[2]<box[3]:
            new_w = int(box[3])
            new_h = int(box[3])
            new_x = int(center_x - 0.5*new_h)
            new_y = int(box[1])
        return [new_x,new_y,new_w,new_h]


def box2square_tight(box):
    if box[2]==box[3]:
        return box
    else:
        center_x=box[0]+0.5*box[2]
        # center_y=box[1]+0.5*box[3]
        center_y=box[1]+0.6*box[3]
        if box[2]<box[3]:
            new_w = int(box[2])
            new_h = int(box[2])
            new_x = int(box[0])
            new_y = int(center_y - 0.5*new_h)
        elif box[2]>box[3]:
            new_w = int(box[3])
            new_h = int(box[3])
            new_x = int(center_x - 0.5*new_h)
            new_y = int(box[1])
        return [new_x,new_y,new_w,new_h]

def box2box_loose(box):
    center_x=box[0]+0.5*box[2]
    center_y=box[1]+0.5*box[3]
    new_w = int(1.1*box[2])
    new_h = int(1.1*box[3])
    new_x = int(center_x - 0.55*box[2])
    new_y = int(center_y - 0.55*box[3])
    return [new_x,new_y,new_w,new_h]

# def main():
#     parser = argparse.ArgumentParser(description='Crop Images by MTCNN',
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--source_dir', help='Directory of to-be-cropped images',type=str,default='../Synthetic_Morph_Dataset/missing/')
#     parser.add_argument('--target_dir', help='Directory of cropped outputs',type=str,default='../Synthetic_Morph_Dataset/missing_MTCNN/')
#     parser.add_argument('--source_suffix', help='Directory of cropped outputs',type=str,default='png')
#     parser.add_argument('--target_suffix', help='Directory of cropped outputs',type=str,default='png')

def main():
    parser = argparse.ArgumentParser(description='Crop Images by MTCNN',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source_dir', help='Directory of to-be-cropped images',type=str,default='../Synthetic_Morph_Dataset/Syn_Dataset_v4/Dev/Female/source_images/')
    parser.add_argument('--target_dir', help='Directory of cropped outputs',type=str,default='../Synthetic_Morph_Dataset/Syn_Dataset_v4/Dev/Female/source_images_MTCNN/')
    parser.add_argument('--source_suffix', help='Directory of cropped outputs',type=str,default='png')
    parser.add_argument('--target_suffix', help='Directory of cropped outputs',type=str,default='png')


    args, _ = parser.parse_known_args()


    source_dir = args.source_dir

    target_dir = args.target_dir
    os.makedirs(target_dir,exist_ok=True)

    source_suffix = args.source_suffix

    image_list = os.listdir(source_dir)

    error_list = []

    for image_name in image_list:
        if image_name.split('.')[-1]==source_suffix:
            try:
                img=PIL.Image.open(os.path.join(source_dir,image_name))
                img_array=np.array(img)
                box=detector.detect_faces(img_array)[0]['box']
                new_box = box
                ROI=img.crop((new_box[0],new_box[1],new_box[0]+new_box[2],new_box[1]+new_box[3]))
                ROI.save(os.path.join(target_dir,image_name[:-3]+'png'),'PNG')
            except:
                error_list.append(image_name)

    with open('MTCNN_log.txt','a+') as f:
        for line in error_list:
            f.writelines(line)
            f.writelines('\n')



if __name__ == "__main__":
    main()