import os
import json
import cv2
import numpy as np
import time
import tensorflow as tf
from scipy import misc
import PIL.Image as Image
import pydensecrf.densecrf as dcrf
from skimage.segmentation import relabel_sequential
import warnings
warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1'


def crop(image_path, coordinate):
    '''
    image_path : Used to read image files - str
    coordinate : Location information that needs to be cropped - [y0,y1,x0,x1]
    cv2.rectangle(image,(left up),(right down),(color),(line width))
    '''
    image = cv2.imread(image_path)
    y0,y1,x0,x1 = coordinate
    img_crop = image[int(y0):int(y1), int(x0):int(x1)]
    return img_crop


def img_crop(sku_info, images_dir, segment_dir, image_wh = None):

    p_xy = sku_info['point_xy']
    bbox = sku_info['bbox']
    file_name = sku_info['file_name']

    if not image_wh:
        image = cv2.imread(os.path.join(images_dir, file_name))
        image_wh = image.shape

    x, y = p_xy[0], p_xy[1]
    w, h = bbox[2], bbox[3]
    # 二倍剪裁
    x0 = (x - w) if (x - w) >= 0 else 0
    y0 = (y - h) if (y - h) >= 0 else 0
    x1 = (x + w) if (x + w) <= image_wh[0] else image_wh[0]
    y1 = (y + h) if (y + h) <= image_wh[1] else image_wh[1]
    nx, ny = (x-x0),(y-y0)
    double_crop_point_xy = [nx,ny]
    coordinate = [y0, y1, x0, x1]
    img_path = os.path.join(images_dir, file_name)
    double_crop_image_path = os.path.join(segment_dir,
                                   img_path.split('/')[-1][:-4] + '_doublecrop.jpg')
    crop_image = crop(img_path, coordinate)
    cv2.imwrite(double_crop_image_path, crop_image)
    #一倍剪裁
    x0, y0, x1, y1 = (x - w/2), (y - h/2), (x + w/2), (y + h/2)
    coordinate = [y0, y1, x0, x1]
    img_path = os.path.join(images_dir, file_name)
    crop_image_path = os.path.join(segment_dir,
                                   img_path.split('/')[-1][:-4] + '_crop.jpg')
    crop_image = crop(img_path, coordinate)
    cv2.imwrite(crop_image_path, crop_image)

    cv2.destroyAllWindows()

    return crop_image_path, double_crop_image_path, double_crop_point_xy


def background_transparent(image_path, trans_image_path):
    img = Image.open(image_path)
    iw,ih = img.size
    img = img.convert('RGBA')
    color_0 = img.getpixel((0,0))
    for h in range(ih):
        for w in range(iw):
            color_1 = img.getpixel((w,h))
            if color_0 == color_1:
                color_1 = color_1[:-1]+(0,)
                img.putpixel((w,h), color_1)
    img.save(trans_image_path)


def create_folder(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except FileNotFoundError as fnfe:
            print(fnfe)
            print('Start creating multi-level directories')
            os.makedirs(path)
        print('Create folder : ',path)
        return 1
    else:
        return 0


def rgba2rgb(img):
    return img[:,:,:3]*np.expand_dims(img[:,:,3],2)


def salience_detection(image_path, segment_dir, gpu_fraction_rate=0.5):

    g_mean = np.array(([126.88, 120.24, 112.19])).reshape([1, 1, 3])
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction_rate)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver = tf.train.import_meta_graph('./meta_graph/my-model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./salience_model'))
        image_batch = tf.get_collection('image_batch')[0]
        pred_mattes = tf.get_collection('mask')[0]
        rgb_pth = image_path
        rgb = misc.imread(rgb_pth)
        if rgb.shape[2] == 4:
            rgb = rgba2rgb(rgb)
        origin_shape = rgb.shape
        rgb = np.expand_dims(
            misc.imresize(rgb.astype(np.uint8), [320, 320, 3], interp="nearest").astype(np.float32) - g_mean, 0)
        feed_dict = {image_batch: rgb}
        pred_alpha = sess.run(pred_mattes, feed_dict=feed_dict)
        final_alpha = misc.imresize(np.squeeze(pred_alpha), origin_shape)
        save_salience_image_path = rgb_pth[:-4]+'_salience'+rgb_pth[-4:]
        misc.imsave(save_salience_image_path, final_alpha)
    return save_salience_image_path


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dense_crf_optimize(image_path, salience_image_path, store_crf_image_path):
    image = cv2.imread(image_path, 1)
    annos = cv2.imread(salience_image_path, 0)
    labels = relabel_sequential(cv2.imread(salience_image_path, 0))[0].flatten()
    output = store_crf_image_path
    EPSILON = 1e-8
    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], M)
    anno_norm = annos / 255.
    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * sigmoid(anno_norm))
    U = np.zeros((M, image.shape[0] * image.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=image, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]
    res = res * 255
    res = res.reshape(image.shape[:2])
    cv2.imwrite(output, res.astype('uint8'))


def cutout(bbox_image_path,crf_crop_image_path,cutout_dir,image_name):
    bbox_image = cv2.imread(bbox_image_path)
    mask_bbox_image = cv2.imread(crf_crop_image_path)
    bbox_height, bbox_width, bbox_channels = bbox_image.shape
    mask_bbox_change_size = cv2.resize(mask_bbox_image, (bbox_width, bbox_height))
    object_image = np.zeros(bbox_image.shape[:3], np.uint8)
    object_image = np.where(mask_bbox_change_size > 50, bbox_image, object_image)
    object_image_name = '%s_%s%s' % (image_name, 'cutout', '.png')
    store_object_image_path = os.path.join(cutout_dir, object_image_name)
    cv2.imwrite(store_object_image_path, object_image)
    return store_object_image_path


def salience2crf(info):
    crf_salience_dir = './train_crop/doublecrop_salience_crf/'
    name = info['file_name'][:-4]
    crop_image_dir = './train_crop/doublecrop/'
    salience_image_dir = './train_crop/doublecrop_salience/'
    crop_image_path = os.path.join(crop_image_dir, name + '_crop.jpg')
    salience_image_path = os.path.join(salience_image_dir, name + '_crop_salience.jpg')
    crf_salience_image_path = os.path.join(crf_salience_dir, name + '_crf.jpg')
    os.system('python examples/dense_hsal.py %s %s %s' % (
        crop_image_path, salience_image_path, crf_salience_image_path))


def dense_crf(double_crop_image_path, salience_image_path, segment_dir, name):
    EPSILON = 1e-8

    img = cv2.imread(double_crop_image_path, 1)
    annos = cv2.imread(salience_image_path, 0)
    labels = relabel_sequential(cv2.imread(salience_image_path, 0))[0].flatten()

    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.
    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    # res *= 255 / res.max()
    res = res * 255
    res = res.reshape(img.shape[:2])
    crf_salience_image_path = os.path.join(segment_dir, name+'_crf.jpg')
    cv2.imwrite(crf_salience_image_path, res.astype('uint8'))
    return crf_salience_image_path


def json2npy(json_path, datatype):

    json_file_path = os.path.join(json_path,'instances_{}2019.json'.format(datatype))
    npy_path = os.path.join(json_path, datatype + '_NIBP')
    if os.path.isfile(npy_path+'.npy'):
        print(f'{npy_path+".npy"} is already exists.')
    else:
        with open(json_file_path, 'r') as load:
            dic = json.load(load)
        start = time.time()
        sku_info = []
        for info in dic['images']:
            name_id = {}
            name_id['file_name'] = info['file_name']
            name_id['id'] = info['id']
            sku_info.append(name_id)
        for idx, info in enumerate(dic['annotations']):
            sku_info[idx]['bbox'] = info['bbox']
            sku_info[idx]['point_xy'] = info['point_xy']
        npy_path = os.path.join(json_path,datatype+'_NIBP')
        np.save(npy_path, sku_info)
        print('f'Complete {"json2npy":^9} in {format(round(time.time() - start, 4):^8} seconds'))


if __name__ == '__main__':
    '''
    1.读取图片和json文件，提取图片信息。
    2.对图片进行剪裁，bbox一倍(crop)和二倍(doublecrop)的剪裁。
    3.对2.中的二倍剪裁(doublecrop)进行显著性检测处理(salience_doublecrop)。
    4.对3.中的结果(salience_doublecrop)进行CRF优化处理(salience_crf_doublecrop)。
    5.对4.中的结果(salience_crf_doublecrop)进行剪裁与2.中的一倍bbox大小匹配(salience_crf_crop)。
    6.用2.中的一倍剪裁(crop)和5.中的结果(salience_crf_crop)进行合成，抠出目标内容(object_image)。
    7.对6.中的结果进行边角透明化处理(object_lim_image)。
    '''

    # 图片存放地址，json文件存放地址
    train_data_path = './data/train2019/' # 训练数据存放地址（单个商品图像）
    json_path = './data/' # json文件存放地址
    # 为了方便使用，提取出json文件中‘file_name''id''bbox''point_xy'四个信息生成npy文件
    image_npy = json2npy(json_path, datatype='train')
    train_sku_info = np.load(r'./data/train_NIBP.npy')
    segment_dir = './train_crop/segment'
    create_folder(segment_dir)

    for i in range(len(train_sku_info)):

        segment_time = time.time()
        info = train_sku_info[i]
        name = info['file_name'][:-4]

        # 对图片进行剪裁 一倍bbox和二倍bbox
        crop_time = time.time()
        crop_image_path, double_crop_image_path, double_crop_point_xy = \
            img_crop(info, train_data_path, segment_dir)
        print(f'Complete {"CROP":^9} in {round(time.time() - crop_time, 4):^8} seconds')

        #对图片进行显著性检测处理
        ssd_time = time.time()
        salience_image_path = salience_detection(double_crop_image_path, segment_dir, 0.6)
        print(f'Complete {"CROP-SSD":^9} in {round(time.time() - ssd_time, 4):^8} seconds')

        #对图片进行CRF细化处理
        crf_time = time.time()
        crf_salience_image_path = dense_crf(double_crop_image_path, salience_image_path, segment_dir, name)
        print(f'Complete {"SSD-CRF":^9} in {round(time.time() - crf_time, 4):^8} seconds')

        # CRF之后进行剪裁
        crf_crop_time = time.time()
        x, y = double_crop_point_xy[0], double_crop_point_xy[1]
        w, h = info['bbox'][2], info['bbox'][3]
        x0, y0, x1, y1 = (x - w / 2), (y - h / 2), (x + w / 2), (y + h / 2)
        crf_crop_img = crop(crf_salience_image_path, [y0,y1,x0,x1])
        crf_crop_img_path = os.path.join(segment_dir, name +'_crf_crop.png')
        cv2.imwrite(crf_crop_img_path, crf_crop_img)
        print(f'Complete {"CRF-CROP":^9} in {round(time.time() - crf_crop_time, 4):^8} seconds')

        #和之前一倍剪裁的图片进行合成，完成cutout
        cutout_time = time.time()
        store_object_image_path = cutout(crop_image_path,crf_crop_img_path,segment_dir,name)
        print(f'Complete {"CROP-SYN":^9} in {round(time.time() - cutout_time, 4):^8} seconds')

        #对cutout处理之后的图片进行背景透明化处理
        trans_time = time.time()
        trans_image_path = os.path.join(segment_dir,name+'_trans.png')
        background_transparent(store_object_image_path, trans_image_path)
        print(f'Complete {"SYN-TRANS":^9} in {round(time.time() - trans_time, 4):^8} seconds')

        print(f'Complete {"SEGMENT":^9} in {round(time.time() - segment_time, 4):^8} seconds')
        break
