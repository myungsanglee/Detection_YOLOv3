'''
yolov3 anchor (k-means clustering) using DarkNet
'''
pascal_voc = {
    "anchors" : [],
    "classes" : 80,
}

coco = {
    "anchors" : [[[116, 90], [156, 198], [373, 326]],
                [[30, 61], [62, 45], [59, 119]],
                [[10, 13], [16, 30], [33, 23]]],
    "classes" : 80,
}