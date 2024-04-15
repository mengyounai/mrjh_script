import uiautomator2 as u2

import numpy as np
import cv2
import time

try:
    print('正在连接...')
    device = u2.connect()
    print('连接成功!')
except:
    print('连接失败')


# stop_app("com.mrjh.tt")
# stop_app("com.amfe.mrjh.vivo")
# stop_app("com.amfe.mrjh.huawei")
# device.app_start("com.mrjh.tt")
# device.click()

def touch(pos):
    device.click(pos[0], pos[1])


# 存在点击
def exist_touch(i):
    if i:
        touch(i)
        return True


# 获取摄像头或视频地址
# cap = cv2.VideoCapture(r"./data/test.mp4")
# 识别置信度阈值

confThreshold = 0.5
# 最大抑制值
nmsThreshold = 0.2
# 网络输入图像的宽度和高度
inpWidth = 320
inpHeight = 320
# coco.names文件存储着80种已经训练好的识别类型名称，并且这些类别名称正好与yolo所训练的80种类别一一对应
classesFile = "./yolov3.names"
# 存储类型名称列表
classNames = []
with open(classesFile, "rt") as f:
    # 依照行读取数据
    classNames = f.read().splitlines()
# 获取手机分辨率
w, h = device.window_size()

# 配置yolov4
modelConfiguration = "./yolov4-tiny-dota2.cfg"  # 配置文件
modelWeights = "./yolov4-tiny-dota2_final.weights"  # 配置权重文件
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)  # 将配置文件加入到dnn网络中
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  # 将DNN后端设置成opencv
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # 将DNN前端设置成cpu驱动


# yolov3检测并处理
def yolo(img):
    # 调整图像大小并转换为网络所需的格式
    image = cv2.imread(img)
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    hT, wT, cT = image.shape  # 获取原始帧图像的大小H,W
    bbox = []  # 创建存储先验框的坐标列表
    classIds = []  # 创建存储每帧检测到的类别信息名称
    confs = []  # 创建每帧读取的置信度值

    for output in outputs:  # 对所有类别遍历
        for det in output:  # 检测frame帧中的每个类别
            scores = det[5:]  # 获取该类别与80项全类别分别的相似概率
            classId = np.argmax(scores)  # 获得80项中最为相似的类别（相似概率值最大的类别）的下标
            confidence = scores[classId]  # 获取最大相似概率的值
            if confidence > confThreshold:  # 判断相似度阈值
                # 获取先验框的四个坐标点
                center_x = int(det[0] * wT)
                center_y = int(det[1] * hT)
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([center_x, center_y, w, h])  # 将坐标添加到bbox中进行存储，便于对frame帧中所有类别的先验框坐标进行存储
                classIds.append(classId)  # 将frame中每一类别对应的编号（1-80），便于在输出文本时，与对应coconame文件中的类别名称进行输出
                confs.append(float(confidence))  # 对frame中识别出来的每一类信息进行最大抑制由参数nms阈值控制
    # 对frame中识别出来的每一类信息进行最大抑制由参数nms阈值控制
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    tagid_lst = []
    tagname = []
    # 初始化坐标列表
    coordinates = []
    for i in indices:
        box = bbox[i]  # 依次读取最大已知参数nms阈值的先验框坐标
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        # 对每个最终识别的目标进行矩形框选
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
        # 对应coco.names相应的类别名称和相似概率进行文字输出
        tagid_lst.append(classIds[i])
        tagname.append(classNames[classIds[i]].capitalize())
        coordinates.append((x, y))
        # cv2.putText(img, f'{classNames[classIds[i]].capitalize()} {int(confs[i] * 100)}%',
        #             (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    # print(f"== 类别id是{tagid_lst} 类别是{tagname}")
    #     scale_percent = 50
    #     new_width = int(image.shape[1] * scale_percent / 100)
    #     new_height = int(image.shape[0] * scale_percent / 100)
    #     resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    #     cv2.imshow("Resized Image", resized_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    if tagid_lst == []:
        return [], [], []
    return tagid_lst, tagname, coordinates


def screenshot():
    device.screenshot("jietu.png")


def attack_js(res):
    print("打僵尸")
    print("res", res)

    if res != ([], [], []):
        if res[1][0] != '僵尸头':
            print("退出")
            return
    else:
        return
    length_of_targets_2 = len(res[2])

    i = 0
    while i < length_of_targets_2:
        touch(res[2][i])
        touch(res[2][i])
        i += 1

    screenshot()
    res = yolo("jietu.png")
    attack_js(res)


def talk_click():
    touch((347, 1053))


#     i = exist_touch(exists(Template(r"tpl1708257200260.png", record_pos=(0.014, 0.789), resolution=(720, 1280))))
#     sleep(3)
#     if i:
#         talk_click()

num = 0

def total_Method():
    global num
    screenshot()
    res = yolo("jietu.png")

    if res != ([], [], []):
        num = 0
        # print("res", res)
        print("目标名", res[1])
        print("坐标", res[2])

        index = 0  # 初始化索引为-1
        for i, item in enumerate(res[1]):
            if '光圈' in item:
                index = i
                break
        touch(res[2][index])
        if res[1] == ['一键升级']:
            device.press('back')
            device.press('back')
            device.press('back')
        if res[1] == ['迎击']:
            time.sleep(10)
        if res[1] == ['生化来袭蓝框']:
            touch((w / 2, h / 2 - 200))
        if res[1][0] == '僵尸头':
            attack_js(res)
    else:
        num += 1
    if num >= 3:
        talk_click()
    time.sleep(2)


while True:
    total_Method()



