import numpy as np
import os
import sys
import cv2
import tensorflow as tf
from darknet import *

flags = tf.app.flags

flags.DEFINE_string('images_dir', '',
                    'Root directory containing images for evaluation. The root '
                    'contains sub-folders of 61 event classes')

flags.DEFINE_string('write_dir', '', 
                    'Root directory to write detection results')

flags.DEFINE_string('meta', '',
                    'darknet metadata')

flags.DEFINE_string('cfg', '',
                    'darknet cfg file')

flags.DEFINE_string('weight', '',
                    'darknet weight')

flags.DEFINE_bool('disp', 'False',
                    'If true, display images with detection boxes.')
#flags.DEFINE_bool('save_image', 'True',
#                    'If true, save images with detection boxes.')

FLAGS = flags.FLAGS


def prepare_filelist(val_path):
    images_to_test = []
    for folder in os.listdir(val_path):
        if os.path.isdir(os.path.join(val_path, folder)):
            sub = os.path.join(val_path, folder)
            for name in os.listdir(sub):
                filepath = os.path.join(sub, name)
                if filepath.endswith('jpg'):
                    cur = {'subfolder': folder, 'basename': name.split('.')[0]}
                    images_to_test.append(cur)
    return images_to_test

def write_detection(write_dir, entry, res_yolo, th_score):
    write_subdir = os.path.join(write_dir, entry['subfolder'])

    if os.path.exists(write_subdir) is False:
        os.makedirs(write_subdir)

    with open(os.path.join(write_subdir, entry['basename'] + '.txt'), 'w') as write_file:
        write_file.write(entry['basename'] + '\n')

        str_to_write = []

        for i in range(0, len(res_yolo)):            
                        
            w = res_yolo[i][2][2]
            h = res_yolo[i][2][3]
            x = res_yolo[i][2][0] - w/2
            y = res_yolo[i][2][1] - h/2
            yolo_score = res_yolo[i][1]

            if yolo_score > th_score:
                str_to_write.append(('%f %f %f %f %f\n' % (x, y, w, h, yolo_score)))
                print x, y, w, h, yolo_score
        
        write_file.write('%d\n' % len(str_to_write))
        
        for s in str_to_write:
            write_file.write(s)

def draw_fd_box(image_np, res_yolo, th_score):

    for i in range(0, len(res_yolo)):            
            
        w = int(res_yolo[i][2][2])
        h = int(res_yolo[i][2][3])
        x = int(res_yolo[i][2][0] - w/2)
        y = int(res_yolo[i][2][1] - h/2)
        yolo_score = res_yolo[i][1]

        if yolo_score > th_score:
            cv2.rectangle(image_np, (x,y), (x+w-1, y+h-1), (0, 0, 255), 4)

    return image_np
    
    

if __name__ == '__main__':
      
    IMAGES_DIR = FLAGS.images_dir
    WRITE_DET_DIR = FLAGS.write_dir
    META_PATH = FLAGS.meta
    CFG_PATH = FLAGS.cfg
    WEIGHT_PATH = FLAGS.weight

    images_to_test = prepare_filelist(IMAGES_DIR)

    net = load_net(CFG_PATH, WEIGHT_PATH, 0)
    meta = load_meta(META_PATH)

    for entry in images_to_test:
        # for image_path in TEST_IMAGE_PATHS:
        image_path = os.path.join(os.path.join(IMAGES_DIR, entry['subfolder']), entry['basename'] + '.jpg')
        print(image_path)

        #detect by yolov2
        r = detect(net, meta, image_path, 0.005)                

        write_detection(WRITE_DET_DIR, entry, r, 0.005)

        if FLAGS.disp:
            image_np = cv2.imread(image_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)    # need this? check performance

            image_result = draw_fd_box(image_np, r, 0.5)
            cv2.imshow("yolo_result", image_result)
            cv2.waitKey(0)
        

    print "Complete!!"


# python detect_on_wider_face_yolo.py --images_dir=/home/ymbaek/Data/wider_face/WIDER_val/images/ --write_dir=/home/ymbaek/Data/wider_face/WIDER_val/predict/ --meta=cfg/widerface.data --cfg=cfg/yolo_face_608.cfg --weight=cfg/yolo_face_270000.weights --disp=True

