import cv2
import numpy as np
import os
import random
from PIL import Image
import time
import math


def choose_sku(sku_dir, mode='easy'):
    """
    :param sku_dir: Folder for storing segmented product images.
    :param mode: Three clutter levels of checkout images.
    :return: A list of selected products.
    """

    sku_list = os.listdir(sku_dir)
    sku_num = len(sku_list)
    if mode == 'easy':
        categories = random.randint(3, 5)
        instsances = random.randint(3, 10)
    elif mode == 'medium':
        categories = random.randint(5, 8)
        instsances = random.randint(10, 15)
    elif mode == 'hard':
        categories = random.randint(8, 10)
        instsances = random.randint(15, 20)
    else:
        print('Please enter the correct mode name : only "easy", "medium" and "hard" are valid.')
        return

    choose_sku_list = []
    if categories >= instsances:
        choose_sku_idx = random.sample(range(0, sku_num), instsances)
        for i in choose_sku_idx:
            choose_sku_path = os.path.join(sku_dir, sku_list[i])
            choose_sku_list.append(choose_sku_path)
    else:
        choose_sku_idx = random.sample(range(0, sku_num), categories)
        for i in choose_sku_idx:
            choose_sku_path = os.path.join(sku_dir, sku_list[i])
            choose_sku_list.append(choose_sku_path)
        for i in range(instsances - categories):
            idx = random.randint(0, len(choose_sku_idx) - 1)
            choose_sku_path = os.path.join(sku_dir, sku_list[choose_sku_idx[idx]])
            choose_sku_list.append(choose_sku_path)
    return choose_sku_list


def background_transparent(image_path, trans_image_path=None):
    """
    :param image_path: The path of the image to be converted.
    :param trans_image_path: Save the path of the converted image.
    :return: Return image if no image is saved.
    """
    img = Image.open(image_path) if isinstance(image_path, str) else image_path
    iw, ih = img.size
    img = img.convert('RGBA')
    color_0 = img.getpixel((0, 0))
    for h in range(ih):
        for w in range(iw):
            color_1 = img.getpixel((w, h))
            if color_0 == color_1:
                color_1 = color_1[:-1] + (0,)
                img.putpixel((w, h), color_1)
    if trans_image_path:
        img.save(trans_image_path)
    else:
        return img


def rotate_image(path, degree):
    """
    :param path: Image path that needs to be rotated.
    :param degree: Angle of rotation.
    :return: Image matrix
    """
    # 此方法只将图像旋转90度以内，通过水平翻转和垂直翻转实现360度旋转的效果。

    # 确定旋转的角度以及翻转的形式。
    angle = degree % 90
    flip_switch = 0
    if angle != 0:
        num = degree % 360 // 90
        if num == 0:
            flip_switch = 0
        else:
            flip_switch = 1
            if num == 1:
                flip_num = 0
            elif num == 2:
                flip_num = -1
            else:
                flip_num = 1
    # 读取图像，确定图像中心，对角线长度及夹角
    image = cv2.imread(path)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    diagonal = (w ** 2 + h ** 2) ** 0.5
    a, b, c = h / 2, w / 2, diagonal / 2
    angle_dia_w = math.degrees(math.acos((a * a - b * b - c * c) / (-2 * b * c)))
    angle_dia_h = 90 - angle_dia_w
    # 计算旋转之后图像所占有的面积，进行放缩使之旋转之后完整成像，并随后进行剪裁使之空白背景最小化。
    ratated_w = round(c * math.cos(math.radians(angle_dia_w - angle)), 2)
    ratated_h = round(c * math.cos(math.radians(angle_dia_h - angle)), 2)
    scale = round(min(w, h) / (diagonal), 2)
    y0, y1 = round(center[1] - ratated_h * scale, 0), round(center[1] + ratated_h * scale, 0)
    x0, x1 = round(center[0] - ratated_w * scale, 0), round(center[0] + ratated_w * scale, 0)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    rotated_cut = rotated[int(y0):int(y1), int(x0):int(x1)]
    # 完成翻转需要后进行格式转换cv2->PIL,以便返回进行背景透明化处理后的图像。
    if flip_switch:
        rotated_flip = cv2.flip(rotated_cut, flip_num)
        rotated_img = Image.fromarray(cv2.cvtColor(rotated_flip, cv2.COLOR_BGR2RGB))
        return background_transparent(rotated_img)
    else:
        rotated_img = Image.fromarray(cv2.cvtColor(rotated_cut, cv2.COLOR_BGR2RGB))
        return background_transparent(rotated_img)


def occlusion_decide(paste_sku_info, pasted_sku_info, threshold):
    """
    :param paste_sku_info: Information of the product image to be pasted.
    :param pasted_sku_info: Information of the product image that has been pasted.
    :param threshold: Coverage threshold.
    :return: Whether the occlusion rate threshold is met, and the information of the latter two images is covered.
    """
    box1 = paste_sku_info['box']
    box2 = pasted_sku_info['box']

    width1 = abs(box1[2] - box1[0])
    height1 = abs(box1[1] - box1[3])
    width2 = abs(box2[2] - box2[0])
    height2 = abs(box2[1] - box2[3])
    x_max = max(box1[0], box1[2], box2[0], box2[2])
    y_max = max(box1[1], box1[3], box2[1], box2[3])
    x_min = min(box1[0], box1[2], box2[0], box2[2])
    y_min = min(box1[1], box1[3], box2[1], box2[3])
    iou_width = x_min + width1 + width2 - x_max
    iou_height = y_min + height1 + height2 - y_max
    if iou_width <= 0 or iou_height <= 0:
        return 1, paste_sku_info, pasted_sku_info
    else:
        occlusion_map1 = paste_sku_info['occlusion_map']
        occlusion_map2 = pasted_sku_info['occlusion_map']
        if box1[0] >= box2[0]:
            # 1 right
            if box1[1] >= box2[1]:
                # 1 down
                for h in range(occlusion_map1.shape[0]):  # 行-深度-h
                    for w in range(occlusion_map1.shape[1]):  # 列-宽度-w
                        if (w <= iou_width) and (h <= iou_height):
                            occlusion_map1[h][w] = 0
                for h in range(occlusion_map2.shape[0]):  # 行-深度-h
                    for w in range(occlusion_map2.shape[1]):  # 列-宽度-w
                        if (w >= box1[0] and (w <= box1[0] + iou_width)) and (
                                h >= box1[1] and (h <= box1[1] + iou_height)):
                            occlusion_map2[h][w] = 0
            else:
                # 1 up
                for h in range(occlusion_map1.shape[0]):  # 行-深度-h
                    for w in range(occlusion_map1.shape[1]):  # 列-宽度-w
                        if (w <= iou_width) and (h >= box2[1] and (h <= box2[1] + iou_height)):
                            occlusion_map1[h][w] = 0
                for h in range(occlusion_map2.shape[0]):  # 行-深度-h
                    for w in range(occlusion_map2.shape[1]):  # 列-宽度-w
                        if (w >= box1[0] and (w <= box1[0] + iou_width)) and (h <= iou_height):
                            occlusion_map2[h][w] = 0
        else:
            # 1 left
            if box1[1] >= box2[1]:
                # 1 down
                for h in range(occlusion_map1.shape[0]):  # 行-深度-h
                    for w in range(occlusion_map1.shape[1]):  # 列-宽度-w
                        if (w >= box2[0] and (w <= box2[0] + iou_width)) and (h <= iou_height):
                            occlusion_map1[h][w] = 0
                for h in range(occlusion_map2.shape[0]):  # 行-深度-h
                    for w in range(occlusion_map2.shape[1]):  # 列-宽度-w
                        if (w <= iou_width) and (h >= box1[1] and (h <= box1[1] + iou_height)):
                            occlusion_map2[h][w] = 0
            else:
                # 1 up
                for h in range(occlusion_map1.shape[0]):  # 行-深度-h
                    for w in range(occlusion_map1.shape[1]):  # 列-宽度-w
                        if (w >= box2[0] and (w <= box2[0] + iou_width)) and (
                                h >= box2[1] and (h <= box2[1] + iou_height)):
                            occlusion_map1[h][w] = 0
                for h in range(occlusion_map2.shape[0]):  # 行-深度-h
                    for w in range(occlusion_map2.shape[1]):  # 列-宽度-w
                        if (w <= iou_width) and (h <= iou_height):
                            occlusion_map2[h][w] = 0
        paste_sku_info['occlusion_map'] = occlusion_map1
        pasted_sku_info['occlusion_map'] = occlusion_map2

        count1, count2 = 0, 0
        area1 = occlusion_map1.shape[0] * occlusion_map1.shape[1]
        area2 = occlusion_map2.shape[0] * occlusion_map2.shape[1]
        for h in range(occlusion_map1.shape[0]):
            for w in range(occlusion_map1.shape[1]):
                if occlusion_map1[h][w] == 0:
                    count1 += 1
        for h in range(occlusion_map2.shape[0]):
            for w in range(occlusion_map2.shape[1]):
                if occlusion_map2[h][w] == 0:
                    count2 += 1

        occlu_rate1 = count1 / area1
        occlu_rate2 = count2 / area2
        if occlu_rate1 > (threshold / 100) or (occlu_rate2 > (threshold / 100)):
            return 0, paste_sku_info, pasted_sku_info
        else:
            return 1, paste_sku_info, pasted_sku_info


def avoid_overlapping(pasted_image, sku, pasted_list, threshold=50):
    """
    :param pasted_image: Background image.
    :param sku: The image to be pasted.
    :param pasted_list: The image that has been pasted.
    :param threshold: Coverage threshold.
    :return:
    """

    # 选择商品图像粘贴位置，以图像左上角坐标位置为基准来粘贴，先排除右侧和下侧的部分位置，保证图像粘贴在背景图像以内。
    bg_w, bg_h = pasted_image.size
    sku_w, sku_h = sku.size
    coordinate_x = random.randint(0, bg_w - sku_w)
    coordinate_y = random.randint(0, bg_h - sku_h)
    coordinate = [coordinate_x, coordinate_y, coordinate_x + sku_w, coordinate_y + sku_h]
    # 制造一个和商品图像形状相同的1值矩阵，并以此来计算遮盖率。
    occlusion_map = np.ones([sku_h, sku_w])  # 图像的宽相当于矩阵的列数
    paste_sku_info = {}
    paste_sku_info['box'] = coordinate
    paste_sku_info['occlusion_map'] = occlusion_map
    if len(pasted_list) == 0:
        return paste_sku_info
    else:
        for idx, pasted_sku_info in enumerate(pasted_list):
            result, paste_sku_info, pasted_sku_infom = occlusion_decide(paste_sku_info, pasted_sku_info, threshold)
            if result:
                pasted_list[idx] = pasted_sku_infom
            else:
                print('paste failed')
                avoid_overlapping(pasted_image, sku, pasted_list, threshold)
        return paste_sku_info


def paste(sku_image, pasted_image, paste_sku_info):
    """
    :param sku_image: The product image to be pasted.
    :param pasted_image: The product image that has been pasted.
    :param paste_sku_info: Image pasted location information.
    :return:
    """
    r, g, b, a = sku_image.split()
    x, y, _, _ = paste_sku_info['box']
    pasted_image.paste(sku_image, (x, y), mask=a)
    return pasted_image


def create_folder(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except FileNotFoundError as error:
            print(error)
            print('Start creating multi-level directories')
            os.makedirs(path)
        print('Create folder : ', path)
        return 1
    else:
        return 0


if __name__ == '__main__':
    '''
    1. 选择商品的种类和数量
    2. 每个商品选择粘贴的位置和角度
    3. 复制粘贴至背景图上即可
    '''

    syn_num = 1  # 设置要合成图像的数量
    paste_sku_info_list = []  # 对合成的商品种类和位置做一个记录，之后用于目标检测。
    for syn_i in range(syn_num):
        syn_time = time.time()
        sku_dir = './train_crop/trans_good/'  # 已经分割处理好的商品图像存放文件夹
        # 1. 选择商品的种类和数量
        choose_sku_list = choose_sku(sku_dir)
        background_image_path = './train_crop/background.jpg'  # 背景图像的地址
        pasted_image = Image.open(background_image_path)
        pasted_list = []  # 保存已经粘贴至背景中的商品图像的信息
        for idx, sku_path in enumerate(choose_sku_list):
            paste_time = time.time()
            # 为要粘贴的商品图像选择位置和角度
            degree = random.randint(1, 359)
            sku_image = rotate_image(sku_path, degree)  # 完成读取商品图像并旋转的工作
            paste_sku_info = avoid_overlapping(pasted_image, sku_image, pasted_list, threshold=25)
            pasted_image = paste(sku_image, pasted_image, paste_sku_info)
            print(f'    Pasting {idx+1} th image takes {round(time.time()-paste_time, 4):<6} seconds')
            pasted_list.append(paste_sku_info)
        paste_sku_info = []
        for idx in range(len(pasted_list)):
            info = {}
            info['file_name'] = choose_sku_list[idx].split('/')[-1][:-10]
            info['bbox'] = pasted_list[idx]['box']
            paste_sku_info.append(info)
        paste_sku_info_list.append(paste_sku_info)
        syn_image_path = './train_crop/syn/' + str(syn_i + 1) + '.jpg'  # 合成图像存放的地址
        pasted_image.save(syn_image_path)
        # print(f'Synthesizing the {syn_i+1} th image takes {round(time.time()-syn_time, 4):<8} seconds')

    syn_info_path = './train_crop/syn/syn_info'
    np.save(syn_info_path, paste_sku_info_list)