import os
import multiprocessing
from multiprocessing.pool import Pool

import numpy as np
import cv2
import time
import json

import uiautomator2
import requests


class role:
    index = None
    is_attack = False
    is_gift = False

    def __str__(self):
        return f"{self.index},{self.is_attack},{self.is_gift}"

    def __init__(self, index):
        self.index = index


def touch(device, pos):
    device.click(pos[0], pos[1])


# 采集相关
def touch_caiji(device, pos, offset=None):
    if offset:
        device.click(pos[0] + offset[0], pos[1] + offset[1])
        return
    device.click(pos[0], pos[1])


# 等待时间  1
wait_time = 0
# 队列是否已满 1
islineMax = False
# 账号数量 1
account_number = 1
# 账号索引 1
account_index = 0
# 角色索引 1
role_index = 1
# 所有角色 1
role_list = []
# 角色数量 1
role_number = 0
# 是否重启 1
is_restart = False
# 采集序列 1
caiji_number = 1
# 奖励序列 1
gift_index = -1

data = {
    "code": "qaq",
    "version": 0,
    "is_gather": 2,
    "type": 0,
    "level": 4,
    "time": "23",
    "is_protect": 2,
    "is_train": 2,
    "is_shuaye": 2,
    "is_switch": 2,
    "is_search": 2,
    "is_sea_monster": 2,
    "is_gift": 0,
    "gift_time": "18",
    "protect_level": 1,
    "main_task": 1
}

code = ''

# 检查文件是否存在
if not os.path.exists("code.txt"):
    with open("code.txt", "w") as file:
        # 可以在此处进行文件的写入操作
        file.write('')


def getCode():
    global code
    try:
        with open("code.txt", "r", encoding="utf-8") as f:
            code = f.readline()
    except (FileNotFoundError, json.JSONDecodeError):
        return


def readConfig(code2):
    global data
    data2 = {'code': code2}

    response = requests.get('http://gasaiyuno.top:8080/readConfig', params=data2)
    # 处理响应
    if response.status_code == 200:
        # 请求成功
        data = json.loads(response.text)


def getConfig_api():
    global data
    try:
        with open("config.txt", "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        readConfig(code)


# 存在点击
def exist_touch(i):
    if i:
        touch(i)
        return True


# 获取摄像头或视频地址
# cap = cv2.VideoCapture(r"./data/test.mp4")
# 识别置信度阈值


getCode()

# getConfig_api()
readConfig(code)

confThreshold = 0.5
# 最大抑制值
nmsThreshold = 0.2
# 网络输入图像的宽度和高度
inpWidth = 320
inpHeight = 320
# coco.names文件存储着80种已经训练好的识别类型名称，并且这些类别名称正好与yolo所训练的80种类别一一对应
modelConfiguration = "./caiji.cfg"  # 配置文件
modelWeights = "./caiji_final.weights"  # 配置权重文件
classesFile = "./caiji.names"
if data['main_task'] == 1:
    classesFile = "./yolov3.names"
    modelConfiguration = "./yolov4-tiny-dota2.cfg"  # 配置文件
    modelWeights = "./yolov4-tiny-dota2_final.weights"  # 配置权重文件
else:
    modelConfiguration = "./caiji.cfg"  # 配置文件
    modelWeights = "./caiji_final.weights"  # 配置权重文件
    classesFile = "./caiji.names"
# 存储类型名称列表
classNames = []
with open(classesFile, "rt") as f:
    # 依照行读取数据
    classNames = f.read().splitlines()

# 配置yolov4
# modelConfiguration = "./yolov4-tiny-dota2.cfg"  # 配置文件
# modelWeights = "./yolov4-tiny-dota2_final.weights"  # 配置权重文件
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


def screenshot(device, name):
    # device.screenshot("jietu.png")
    device.screenshot(name)


def attack_js(device, res, name):
    print("打僵尸")
    print(f"res: {res}")
    if res != ([], [], []):
        if res[1][0] != '僵尸头':
            print("退出")
            return
    else:
        return
    length_of_targets_2 = len(res[2])

    i = 0
    while i < length_of_targets_2:
        touch(device, res[2][i])
        touch(device, res[2][i])
        i += 1

    screenshot(device, name)
    # res = yolo("jietu.png")
    res = yolo(name)
    attack_js(device, res, name)


def talk_click(device):
    touch(device, (347, 1053))

num = 0


def total_Method(device_name):
    global num
    device = uiautomator2.connect(device_name)
    print("device_name:", device_name)
    name = device_name + "-jietu.png"
    print("name:", name)

    while True:
        screenshot(device, name)
        res = yolo(name)

        # 获取手机分辨率
        w, h = device.window_size()

        if res != ([], [], []):
            num = 0
            # print("res", res)
            print(f"{device_name}目标名: {res[1]}")
            print(f"{device_name}坐标: {res[2]}")

            index = 0  # 初始化索引为-1
            for i, item in enumerate(res[1]):
                if '光圈' in item:
                    index = i
                    break
            touch(device, res[2][index])
            if res[1] == ['一键升级']:
                device.press('back')
                device.press('back')
                device.press('back')
            if res[1] == ['迎击']:
                time.sleep(10)
            if res[1] == ['生化来袭蓝框']:
                touch(device, (w / 2, h / 2 - 200))
            if res[1][0] == '僵尸头':
                attack_js(device, res, name)
        else:
            num += 1
        if num >= 3:
            talk_click(device)
        time.sleep(2)


def highest_line(device, name):
    screenshot(device, name)
    res = yolo(name)
    # res = yolo("jietu.png")
    if repeat_operate(device, res, isClick=True, first='跳过引导'):
        print("点击跳过引导")
        return True
    if repeat_operate(device, res, isClick=True, first='关闭框'):
        print("点击关闭")
        return True
    if repeat_operate(device, res, isClick=True, first='重建'):
        print("点击重建")
        return True
    if repeat_operate(device, res, isClick=True, first='稍后在说'):
        print("点击稍后在说")
        return True
    return False


def extra(device, name):
    touch_caiji(device, (18, 637))
    # 点击箭头
    # common_operate(device,name,"任务栏")
    time.sleep(2)
    if data['is_train'] == 2:
        print("开始造兵")
        new_train(device, name)

    swipe(device, (300, 300), (300, 50))
    time.sleep(2)

    if data['is_search'] == 2:
        print("开始寻宝")
        search(device, name)

    time.sleep(2)

    if data['is_sea_monster'] == 2:
        print("猎杀海怪")
        sea_monster(device, name)




    touch_caiji(device, (63, 1236))


def sea_monster(device,name):
    offset = (165, 40)
    # 已打完
    common_operate(device,name,"海怪领奖", offset=offset)
    time.sleep(2)
    # 还有次数
    common_operate(device,name,"猎杀海怪", offset=offset)
    i = True
    if i:
        time.sleep(2)
        print("立即加入2")
        j = common_operate(device,name,"立即加入")
        # touch_caiji(device, (537, 769))

        common_operate(device,name,"立即加入")
        # touch_caiji(device,(590, 404))


def search(device, name):
    offset = (171, -20)
    # 领奖
    i = common_operate(device, name, "寻宝领奖", offset=offset)
    if i:
        time.sleep(2)
        common_operate(device, name, "收取")
        time.sleep(2)
        common_operate(device, name, "领取")
        device.press('back')

    # 还有次数
    i = common_operate(device, name, "寻宝", offset=offset)
    if i:
        z = common_operate(device, name, "收取", isClick=False)
        if z:
            print("返回")
            touch_caiji(device, (62, 76))
            return
        time.sleep(2)
        j = common_operate(device, name, "选择执行官")
        if j:
            time.sleep(2)
            common_operate(device, name, "前往")
            time.sleep(2)
            common_operate(device, name, "完成")

        common_operate(device, name, "搜索")
        touch_caiji(device, (62, 76))


def new_train(device, name):
    offset = (300, 0)
    print("点击步兵营")
    train_operate(device, name, common_operate(device, name, "步兵营", offset=offset))

    print("点击射击场")
    train_operate(device, name, common_operate(device, name, "射击场", offset=offset))
    print("点击车辆改造场")
    train_operate(device, name, common_operate(device, name, "车辆改造场", offset=offset))


def train_operate(device, name, bool):
    time.sleep(2)
    i = common_operate(device, name, first="一键加速", isClick=False)
    if i:
        print("点击返回")
        time.sleep(2)
        common_operate(device, name, "返回")
        return

    if not bool:
        return
    touch_caiji(device, (82, 912))
    touch_caiji(device, (220, 1082))
    touch_caiji(device, (496, 1080))
    touch_caiji(device, (584, 1212))
    time.sleep(2)


def common_operate(device, name, first=None, isClick=True, offset=None):
    screenshot(device, name)
    res = yolo(name)
    return repeat_operate(device,res, first=first, isClick=isClick, offset=offset)


def repeat_operate(device, res, isClick=True, first=None, offset=None):
    found_match = False
    if res != ([], [], []):
        # print("res", res)
        print(f"{device.serial}目标名: {res[1]}")
        print(f"{device.serial}坐标: {res[2]}")
        if first:
            for i, item in enumerate(res[1]):
                if first in item:
                    found_match = True
                    if isClick:
                        print("点击", first)
                        print("点击坐标为", res[2][i])

                        touch_caiji(device, res[2][i], offset)
                        return found_match
        if isClick and first is None:
            touch_caiji(device, res[2][0])
    return found_match


def new_protect(device,name):
    print("开罩")
    time.sleep(2)
    # 点击总部
    w, h = device.window_size()
    touch_caiji(device, (h / 2, w / 2))

    time.sleep(2)
    # wait_thing(device,name,"搜寻", 10, True)
    print("城市增益")
    # i = common_operate(device,name,"城市增益")
    touch_caiji(device,(224, 649))
    i = True
    time.sleep(2)
    if i:
        print("停战协议")
        j = common_operate(device,name,"停战协议")
        time.sleep(5)
        if j:
            if data['protect_level'] == 0:
                touch_caiji(device, (641, 439.72))
                protect_solve(device,name)

            if data['protect_level'] == 1:
                touch_caiji(device, (641, 623.16))
                protect_solve(device,name)

            if data['protect_level'] == 2:
                touch_caiji(device, (641, 807))
                protect_solve(device,name)

            if data['protect_level'] == 3:
                touch_caiji(device, (641, 950))
                protect_solve(device,name)

    device.press('back')
    device.press('back')


def click_image(device, image_path, timeout=10):
    start_time = time.time()
    w, h = device.window_size()
    while True:
        current_time = time.time()
        if current_time - start_time > timeout:
            print("Timeout")
            return False

        try:
            img = cv2.imread(image_path, 0)
            img_width, img_height = img.shape[::-1]
            screen = device.screenshot(format='opencv')
            result = cv2.matchTemplate(screen, img, cv2.TM_CCOEFF_NORMED)
            # result = cv2.cvtColor(screen,img, cv2.COLOR_BGR2RGB)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if max_val > 0.8:
                x, y = max_loc[0] + img_width / 2, max_loc[1] + img_height / 2
                print("x,y坐标", x, y)
                device.click(x / w, y / h)
                return True
        except Exception as e:
            print(e)

        time.sleep(1)


def switch_account(device, name):
    global role_index, account_index, is_restart, gift_index

    time.sleep(10)

    print("重启方式判断")

    # 是否重启方式登录
    if not is_restart:
        print("正常切号")
        touch_caiji(device, (59, 99))

        swipe(device, (430, 1188), (157, 1188))

        time.sleep(2)
        common_operate(device, name, "设定")
        time.sleep(2)
        common_operate(device, name, "切换账号")
        time.sleep(2)
        common_operate(device, name, "黄色确定")

    print("异常重启")
    is_restart = False

    print("绿色进入游戏")
    i = common_operate(device, name, "绿色进入游戏")
    if not i:
        print("寻找黄色进入游戏")
        common_operate(device, name, "黄色进入游戏")
        touch_caiji(device,(360, 1100))
        time.sleep(10)
        return

    # 选择角色
    wait_thing(device,name,"黄色进入游戏", 10)
    common_operate(device,name,"变更")

    role_name = str(role_index) + ".png"

    click_image(device, f"./image/{role_name}")
    # 是否切换账号
    if role_index == int(account_number):
        role_index = 1
        account_index += 1
    else:
        role_index += 1

    common_operate(device,name,"黄色进入游戏")

    if len(role_list) < role_number:
        ro = role(role_name)
        role_list.append(ro)

    gift_index = gift_index + 1


# 确保在基地
def wait_thing(device, name, first, sleepNum, isClick=False):
    global wait_time
    screenshot(device, name)
    res = yolo(name)
    print("res", res)
    if res != ([], [], []):
        print("目标名", res[1])
        print("坐标", res[2])
        for i, item in enumerate(res[1]):
            if first in item:
                wait_time = 0
                if isClick:
                    touch_caiji(device, res[2][i])
                return True
    time.sleep(1)
    wait_time += 1
    print("wait_time", wait_time)
    if wait_time >= sleepNum:
        print("超时")
        restart(device,name)
        wait_time = 0
        return False
    wait_thing(device, name, first, sleepNum, isClick)


def protect_solve(device, name):
    time.sleep(2)
    i = common_operate(device, name, "立即加入")
    if i:
        return


def new_shuaye(device, name):
    print("打野")
    time.sleep(3)
    touch_caiji(device,(51, 475))
    # wait_thing(device, name, "搜寻", 10, True)
    # common_operate(device,name,"野怪")
    time.sleep(2)
    touch_caiji(device, (120, 1000))
    time.sleep(3)

    touch_caiji(device, (358, 1233))
    time.sleep(2)
    print("发动进攻")
    i = common_operate(device, name, "发动进攻", isClick=True)
    if i:
        time.sleep(2)
        common_operate(device, name)
    time.sleep(2)

    lineMaxErrorHandler(device, name)

    i = attack(device, name)

    if i:
        j = power_not_enough(device, name)
        if j:
            return
        return
    return


def lineMaxErrorHandler(device, name):
    global islineMax
    i = common_operate(device, name, "黄色确定")
    islineMax = False
    if i:
        islineMax = True
        return True

    print("队列未满")
    return False


def power_not_enough(device, name):
    i = common_operate(device, name, "继续进攻")
    if i:
        return True
    return False


def nobinErrorHander(device, name):
    time.sleep(2)
    i = common_operate(device, name, "黄色确定")
    if not i:
        i = common_operate(device, name, "跳过引导")
    if i:
        touch_caiji(device, (62, 76))
        return False
    return True


def attack(device, name):
    global islineMax
    if not islineMax:
        time.sleep(2)
        touch_caiji(device, (584, 1212))
        return nobinErrorHander(device, name)
    print("出兵失败!")
    return False


def add(device):
    touch_caiji(device,(467, 1152))


def reduce(device):
    touch_caiji(device, (44, 1152))


def tiaozheng_level(device):
    print("调整采集等级")
    i = 1
    j = 1
    while i <= 9:
        reduce(device)
        i += 1
    while j <= data['level']:
        add(device)
        j += 1


def swipe(device, pos, topos):
    device.swipe(pos[0], pos[1], topos[0], topos[1], duration=0.05)


# 0:水，1：罐头，2：石油，3：矿石，4：水罐混采，5：油矿混采
def caiji(device, name, type):
    global caiji_number, type_rand
    print("采集")
    if type == 4:
        type_rand = caiji_number % 2
        caiji_number = caiji_number + 1
    elif type == 5:
        type_rand = caiji_number % 2 + 2
        caiji_number = caiji_number + 1
    else:
        type_rand = type

    time.sleep(3)
    touch(device,(51, 475))
    # wait_thing(device, name, "搜寻", 10, True)
    time.sleep(2)

    swipe(device, (576, 1005), (320, 1005))
    time.sleep(2)
    # 罐头
    if type_rand == 1:
        print("采集罐头")
        touch_caiji(device, (260, 1004))

        # 石油
    elif type_rand == 2:
        print("采集石油")
        touch_caiji(device, (516, 1000))


    # 水
    elif type_rand == 0:
        print("水")
        touch_caiji(device, (386, 999))


    # 矿石
    elif type_rand == 3:
        print("采集矿石")
        touch_caiji(device, (637, 999))

    tiaozheng_level(device)
    touch_caiji(device, (358, 1233))

    time.sleep(2.0)

    touch_caiji(device, (490, 696))

    lineMaxErrorHandler(device, name)

    i = attack(device, name)
    if i:
        print("还有队列")
        caiji(device, name, type)
        return
    print("采集流程结束")
    return


def caiji_total(device_name):
    device = uiautomator2.connect(device_name)
    print("device_name:", device_name)
    name = device_name + "-jietu.png"
    # print("name:", name)
    while True:
        try:
            highest_line(device, name)
            # i = common_operate(device,name,"世界",False)
            i = wait_thing(device,name,"任务栏",10,False)
            if not i:
                continue
            extra(device, name)
            time.sleep(2)
            # 进入世界
            touch_caiji(device, (63, 1236))

            i = highest_line(device, name)
            if i:
                time.sleep(5)
            if data['is_protect'] == 2:
                new_protect(device, name)
                # 刷野
            if data['is_shuaye'] == 2:
                new_shuaye(device, name)
                # 采集
            if data['is_gather'] == 2:
                caiji(device, name, data['type'])

            touch_caiji(device, (63, 1236))
            time.sleep(int(data['time']))

            common_operate(device, name)

        except:
            restart(device,name)

# 重启
def restart(device,name):
    global is_restart

    print("重启")

    device.keyevent("home")

    # 0:官网，1:华为,2:vico
    # 关闭应用
    if data['version'] == 0:
        device.app_stop("com.mrjh.tt")
    elif data['version'] == 1:
        device.app_stop("com.amfe.mrjh.huawei")
    elif data['version'] == 2:
        device.app_stop("com.amfe.mrjh.vivo")

    time.sleep(3)

    if data['version'] == 0:
        device.app_start("com.mrjh.tt")
    elif data['version'] == 1:
        device.app_start("com.amfe.mrjh.huawei")
    elif data['version'] == 2:
        device.app_start("com.amfe.mrjh.vivo")

    time.sleep(10)
    is_restart = True
    switch_account(device,name)

def get_devices():
    # 多设备检测
    # 1、执行adb命令
    str = os.popen("adb devices").readlines()
    print(str)
    device_info = []
    # 2、循环遍历列表
    for i in str[1:]:
        # 排除第一行，查找行中含有device关键字
        if "device" in i and i != "List of devices attached \n":
            device_name = i.replace('\tdevice\n', '')
            device_info.append(device_name)
    return device_info




# devices = get_devices()
# for device in devices:
#     print("device", device)
#     if data['main_task'] == 1:
#         print("开启主线")
#         process = multiprocessing.Process(target=total_Method, args=(device,))
#     else:
#         print("开启采集")
#         process = multiprocessing.Process(target=caiji_total, args=(device,))
#     # 启动进程
#     process.start()

if __name__ == '__main__':
    # 获取所有已连接的设备名
    devices = get_devices()

    multiprocessing.freeze_support()

    for device in devices:
        print("device", device)
        if data['main_task'] == 1:
            print("开启主线")
            process = multiprocessing.Process(target=total_Method, args=(device,))
        else:
            print("开启采集")
            process = multiprocessing.Process(target=caiji_total, args=(device,))

        # 启动进程
        process.start()


