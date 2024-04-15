import json
import os
import time

import cv2
import numpy as np
import requests
import uiautomator2 as u2

try:
    print('正在连接...')
    device = u2.connect()
    print('连接成功!')
except:
    print('连接失败')

w, h = device.window_size()


def click_image(image_path, timeout=10):
    start_time = time.time()

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


def touch(pos, offset=None):
    if offset:
        device.click(pos[0] + offset[0], pos[1] + offset[1])
        return
    device.click(pos[0], pos[1])


confThreshold = 0.5
# 最大抑制值
nmsThreshold = 0.2
# 网络输入图像的宽度和高度
inpWidth = 320
inpHeight = 320
# coco.names文件存储着80种已经训练好的识别类型名称，并且这些类别名称正好与yolo所训练的80种类别一一对应
classesFile = "./caiji.names"
# 存储类型名称列表
classNames = []
with open(classesFile, "rt") as f:
    # 依照行读取数据
    classNames = f.read().splitlines()

# 配置yolov4
modelConfiguration = "./caiji.cfg"  # 配置文件
modelWeights = "./caiji_final.weights"  # 配置权重文件
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

    if tagid_lst == []:
        return [], [], []
    return tagid_lst, tagname, coordinates


def screenshot():
    device.screenshot("jietu.png")


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
    "is_train": 1,
    "is_shuaye": 1,
    "is_switch": 2,
    "is_search": 2,
    "is_sea_monster": 2,
    "is_gift": 0,
    "gift_time": "18",
    "protect_level": 0
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


def getUserNumber():
    global account_number
    count = 0

    files = os.listdir(os.getcwd() + "/image")

    # 遍历文件列表
    for file in files:
        if file.endswith(".png"):  # 检查文件名是否以 ".png" 结尾
            count += 1  # 如果是，增加计数器的值

    if count >= 3:
        count = 3
    account_number = count


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


class role:
    index = None
    is_attack = False
    is_gift = False

    def __str__(self):
        return f"{self.index},{self.is_attack},{self.is_gift}"

    def __init__(self, index):
        self.index = index


def back():
    device.press('back')


def lineMaxErrorHandler():
    global islineMax
    i = common_operate("黄色确定")
    islineMax = False
    if i:
        islineMax = True
        return True

    print("队列未满")
    return False


def power_not_enough():
    i = common_operate("继续进攻")
    if i:
        return True
    return False


def nobinErrorHander():
    time.sleep(2)
    i = common_operate("黄色确定")
    if not i:
        i = common_operate("跳过引导")
    if i:
        touch((62, 76))
        return False
    return True


def attack():
    global islineMax
    if not islineMax:
        time.sleep(2)
        touch((584, 1212))
        return nobinErrorHander()
    print("出兵失败!")
    return False


def add():
    touch((467, 1152))


def reduce():
    touch((44, 1152))


def tiaozheng_level():
    i = 1
    j = 1
    while i <= 9:
        reduce()
        i += 1
    while j <= data['level']:
        add()
        j += 1


def swipe(pos, topos):
    device.swipe(pos[0], pos[1], topos[0], topos[1], duration=0.05)


# 0:水，1：罐头，2：石油，3：矿石，4：水罐混采，5：油矿混采
def caiji(type):
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

    wait_thing("搜寻", 2, True)
    time.sleep(2)

    swipe((576, 1005), (320, 1005))
    time.sleep(2)
    # 罐头
    if type_rand == 1:
        print("采集罐头")
        touch((260, 1004))

        # 石油
    elif type_rand == 2:
        print("采集石油")
        touch((516, 1000))


    # 水
    elif type_rand == 0:
        print("水")
        touch((386, 999))


    # 矿石
    elif type_rand == 3:
        print("采集矿石")
        touch((637, 999))

    tiaozheng_level()
    touch((358, 1233))

    time.sleep(2.0)

    touch((490, 696))

    lineMaxErrorHandler()

    i = attack()
    if i:
        print("还有队列")
        caiji(type)
        return
    print("采集流程结束")
    return


def switch_account():
    global role_index, account_index, is_restart, gift_index

    # 是否重启方式登录
    if not is_restart:
        touch((59, 99))

        swipe((430, 1188), (157, 1188))

        time.sleep(2)
        common_operate("设定")
        time.sleep(2)
        common_operate("切换账号")
        time.sleep(2)
        common_operate("黄确定")

    is_restart = False

    time.sleep(5)
    i = common_operate("绿色进入游戏")
    if not i:
        common_operate("黄色进入游戏")
        return

    # 选择角色
    wait_thing("黄色进入游戏", 2)
    common_operate("变更")

    role_name = str(role_index) + ".png"

    click_image(f"./image/{role_name}")
    # 是否切换账号
    if role_index == int(account_number):
        role_index = 1
        account_index += 1
    else:
        role_index += 1

    common_operate("黄色进入游戏")

    if len(role_list) < role_number:
        ro = role(role_name)
        role_list.append(ro)

    gift_index = gift_index + 1


# 重启
def restart():
    global is_restart
    # i = wait_thing("世界", 1)

    # if i:
    #     return
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
    switch_account()


def protect_solve():
    time.sleep(2)
    i = common_operate("立即加入")
    if i:
        return
    # common_operate()


def new_protect():
    print("开罩")
    time.sleep(2)
    # 点击总部
    touch((h / 2, w / 2))
    # touch((w / 2, h / 2))
    wait_thing("搜寻", 2, False)
    print("城市增益")
    i = common_operate("城市增益")
    time.sleep(2)
    if i:
        print("停战协议")
        j = common_operate("停战协议")
        time.sleep(5)
        if j:
            if data['protect_level'] == 0:
                touch((641, 439.72))
                protect_solve()

            if data['protect_level'] == 1:
                touch((641, 623.16))
                protect_solve()

            if data['protect_level'] == 2:
                touch((641, 807))
                protect_solve()

            if data['protect_level'] == 3:
                touch((641, 950))
                protect_solve()

    back()
    back()


def train_operate(bool):
    time.sleep(2)
    i = common_operate("一键加速", isClick=False)
    if i:
        print("点击返回")
        time.sleep(2)
        common_operate("返回")
        return

    if not bool:
        return
    touch((82, 912))
    touch((220, 1082))
    touch((496, 1080))
    touch((584, 1212))
    time.sleep(2)


def new_train():
    offset = (300, 0)
    print("点击步兵营")
    train_operate(common_operate("步兵营", offset=offset))

    print("点击射击场")
    train_operate(common_operate("射击场", offset=offset))
    print("点击车辆改造场")
    train_operate(common_operate("车辆改造场", offset=offset))
    # print("点击生化人工厂")
    # i = common_operate("生化人工厂", offset=offset)
    # if i:
    #     time.sleep(3)
    #     swipe((h/2,100),(h/2,w/2))
    #     j = common_operate()
    #     if j:
    #         touch((82, 912))
    #         touch((220, 1082))
    #         touch((496, 1080))
    #         touch((584, 1212))
    #         time.sleep(2)
    #     else:
    #         touch((62, 76))


def sea_monster():
    offset = (200, 0)
    # 已打完
    common_operate("海怪领奖", offset=offset)
    # 还有次数
    i = common_operate("猎杀海怪", offset=offset)
    if i:
        time.sleep(5)
        j = common_operate("立即加入")
        if j:
            print("邀请盟友")
            return
        common_operate("立即加入")


def search():
    offset = (200, 0)
    # 领奖
    i = common_operate("寻宝领奖", offset=offset)
    if i:
        time.sleep(2)
        common_operate("收取")
        time.sleep(2)
        common_operate("领取")
        back()

    # 还有次数
    i = common_operate("寻宝", offset=offset)
    if i:
        z = common_operate("收取", isClick=False)
        if z:
            print("返回")
            touch((62, 76))
            return
        time.sleep(2)
        j = common_operate("选择执行官")
        if j:
            time.sleep(2)
            common_operate("前往")
            time.sleep(2)
            common_operate("完成")

        common_operate("搜索")
        touch((62, 76))


def new_shuaye():
    wait_thing("搜寻", 2, True)
    # common_operate("野怪")
    touch((100, 1000))
    time.sleep(3)

    touch((358, 1233))
    time.sleep(2)
    print("发动进攻")
    i = common_operate("发动进攻", isClick=True)
    if i:
        common_operate()
    time.sleep(2)

    lineMaxErrorHandler()

    i = attack()

    if i:
        j = power_not_enough()
        if j:
            return
        return
    return


def extra():
    touch((18, 637))
    # 点击箭头
    # common_operate("任务栏")
    time.sleep(2)
    if data['is_train'] == 2:
        print("开始造兵")
        new_train()

    swipe((300, 300), (300, 50))
    time.sleep(2)

    if data['is_sea_monster'] == 2:
        print("猎杀海怪")
        sea_monster()

    time.sleep(2)
    if data['is_search'] == 2:
        print("开始寻宝")
        search()

    touch((63, 1236))


# getCode()

# getConfig_api()
# getConfig_extra()

getUserNumber()

print("用户数量", account_number)


def highest_line():
    screenshot()
    res = yolo("jietu.png")
    if repeat_operate(res, isClick=True, first='跳过引导'):
        print("点击跳过引导")
        return True
    if repeat_operate(res, isClick=True, first='关闭框'):
        print("点击关闭")
        return True
    if repeat_operate(res, isClick=True, first='重建'):
        print("点击重建")
        return True
    if repeat_operate(res, isClick=True, first='稍后在说'):
        print("点击稍后在说")
        return True
    return False


def repeat_operate(res, isClick=True, first=None, offset=None):
    if res != ([], [], []):
        # print("res", res)
        print("目标名", res[1])
        print("坐标", res[2])
        if first:
            for i, item in enumerate(res[1]):
                if first in item:
                    # print("isClick:", isClick)
                    if isClick:
                        print("点击", first)
                        print("点击坐标为", res[2][i])
                        touch(res[2][i], offset)
                        return True
        if isClick and first is None:
            touch(res[2][0])
    return False


def common_operate(first=None, isClick=True, offset=None):
    screenshot()
    res = yolo("jietu.png")
    return repeat_operate(res, first=first, isClick=isClick, offset=offset)


# 确保在基地
def wait_thing(name, sleepNum, isClick=False):
    global wait_time
    screenshot()
    res = yolo("jietu.png")
    print("res", res)
    if res != ([], [], []):
        print("目标名", res[1])
        print("坐标", res[2])
        for i, item in enumerate(res[1]):
            if name in item:
                wait_time = 0
                if isClick:
                    touch(res[2][i])
                return True
    time.sleep(5)
    wait_time += 1
    print("wait_time", wait_time)
    if wait_time >= sleepNum:
        print("超时")
        # restart()
        return False
    wait_thing(name, sleepNum, isClick)

# touch((460, 660))
print(h/2)
# (295, 622)
# (145, 355), (135, 250), (134, 191), (149, 315)
# while True:
#     try:
#         highest_line()
#         # wait_thing("世界", 3)
#         extra()
#         time.sleep(2)
#         # 进入世界
#         touch((63, 1236))
#
#         i = highest_line()
#         if i:
#             time.sleep(5)
#         if data['is_protect'] == 2:
#             new_protect()
#             # 刷野
#         if data['is_shuaye'] == 2:
#             new_shuaye()
#             # 采集
#         if data['is_gather'] == 2:
#             caiji(data['type'])
#
#         touch((63, 1236))
#         time.sleep(int(data['time']))
#
#         # 切号
#         # if data['is_switch'] == 2:
#         #     if account_number != 0:
#         #         switch_account()
#         common_operate()
#
#     except:
#         print("重启")
#         # restart()
