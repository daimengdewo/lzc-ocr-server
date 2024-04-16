from fastapi import FastAPI
from paddleocr import PaddleOCR
from pydantic import BaseModel
import time
import uvicorn
import re

app = FastAPI()
ocrName = "ocr/"


class IdCardImages(BaseModel):
    idCardFront: str
    idCardBack: str


# 指定检测模型和识别模型路径
det_model_dir = '.\\det_model'
rec_model_dir = '.\\rec_model'
cls_model_dir = '.\\cls_model'


# ocr
@app.post("/{}{}".format(ocrName, "id_card"))
async def ocrIdCard(id_card_images: IdCardImages):
    PaddleOCR(det_model_dir=det_model_dir, rec_model_dir=rec_model_dir, cls_model_dir=cls_model_dir)
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False, use_mp=True)
    idCardFront = id_card_images.idCardFront
    idCardBack = id_card_images.idCardBack
    img_paths = [idCardFront, idCardBack]
    dataList = []
    start_time = time.time()
    for img_path in img_paths:
        result = ocr.ocr(img_path, cls=True)
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                data = line[1][0]
                Str = getInformation(data)
                if Str != '':
                    dataList.append(Str)

    dataList = pretreatment(dataList)
    resultDict = findResultReserve(dataList)
    end_time = time.time()
    total_time = end_time - start_time

    return {"data": resultDict, "msg": "总耗时：{} 秒".format(total_time), "test": dataList}


def getInformation(data):
    filtered_data = removePY(data)
    NoneSpaceStr = removeSpace(filtered_data)
    NonePunctuationStr = removePunctuation(NoneSpaceStr)
    return NonePunctuationStr


def removePY(data):
    filtered_data = ''
    idx = 0
    while idx < len(data):
        char = data[idx]
        if char.upper() in 'QWERTYUIOPASDFGHJKLZXCVBNM':
            # 如果是英文字母但不是大写字母表中的字母
            if idx == 17 and data[idx - 17:idx].isdigit():
                # 如果前面有17位数字，则保留
                filtered_data += char
            else:
                # 如果前面不是17位数字，则跳过
                pass
            idx += 1
        else:
            # 如果不是英文字母，保留
            filtered_data += char
            idx += 1
    return filtered_data


def removeSpace(long_str):
    noneSpaceStr = ''
    str_arry = long_str.split()
    for x in range(0, len(str_arry)):
        noneSpaceStr = noneSpaceStr + str_arry[x]
    return noneSpaceStr


def removePunctuation(noneSpaceStr):
    punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！『【】（）、。：；’‘……￥·"""
    s = noneSpaceStr
    dicts = {i: '' for i in punctuation}
    punc_table = str.maketrans(dicts)
    nonePunctuationStr = s.translate(punc_table)
    return nonePunctuationStr


def getYxqx(item, keyword, info_dict):
    # 第一个有效期限
    date_str1 = item.replace(keyword, "").strip()[:8]
    formatted_date1 = f"{date_str1[:4]}-{date_str1[4:6]}-{date_str1[6:8]}"
    info_dict["有效期限1"] = formatted_date1
    # 第二个有效期限
    date_str2 = item.replace(keyword, "").strip()[8:16]
    formatted_date2 = f"{date_str2[:4]}-{date_str2[4:6]}-{date_str2[6:8]}"
    info_dict["有效期限2"] = formatted_date2
    return info_dict


def findResultReserve(data):
    # 定义正则表达式模式
    patterns = {
        '姓名': r'.*姓名(.*)$',
        '性别': r'.*性别(.*)$',
        '民族': r'.*民族(.*)$',
        '住址': r'.*住址(.*)$',
        '签发机关': r'.*签发机关(.*)$',
        '有效期限': r'.*有效期限(.*)$',
        '公民身份号码': r'.*公民身份号码(.*)$',
        "出生": r'.*出生(.*)$'
    }

    patterns2 = {
        '出生': r'\d{4}年\d{1,2}月\d{1,2}日',
        '签发机关': r'.*公安局.*',
        '有效期限': r'\d{16}',
        '公民身份号码': r'\d{17,}'
    }

    # 创建一个空字典用于存储信息
    info_dict = {}

    # 遍历数据列表
    for item in data:
        # 遍历正则表达式模式字典
        for key, pattern in patterns.items():
            # 使用正则表达式匹配对应项
            match = re.match(pattern, item)
            if match:
                if key == '有效期限':
                    info_dict = getYxqx(item, key, info_dict)
                    break
                else:
                    # 如果匹配成功，截断这个项，然后将信息存储到字典中
                    info = match.group(1)
                    info_dict[key] = info
                if key == "住址":
                    addrs = getAddr(data)
                    for addr in addrs:
                        info_dict["住址"] = info_dict["住址"] + addr
                if key == "公民身份号码":
                    info_dict["公民身份号码"] = find_long_number(data)
                break  # 匹配成功后跳出内层循环
        # 遍历正则表达式模式字典2
        for key2, pattern2 in patterns2.items():
            # 使用正则表达式匹配对应项
            match = re.match(pattern2, item)
            if match:
                if key2 == '有效期限':
                    info_dict = getYxqx(item, key2, info_dict)
                    break
                else:
                    # 如果匹配成功，直接将信息存储到字典中
                    info_dict[key2] = item
                break  # 匹配成功后跳出内层循环

    return info_dict


def getAddr(data):
    keywords = ["姓名", "民族", "性别", "出生", "住址", "签发机关", "有效期限",
                "公民身份号码", "中华人民共和国", "居民身份证", "公安局"]
    keywords2 = ["中国"]

    filtered_data = [item for item in data if (
            not any(keyword in item for keyword in keywords) and
            item not in keywords2 and
            not (len(item) > 15 and item.isdigit()) and
            not re.match(r'\d{4}年\d{1,2}月\d{1,2}日', item)
    )]

    return filtered_data


def find_long_number(lst):
    for item in lst:
        if len(item) > 17 and item.isdigit():
            return item


def pretreatment(data):
    keywords = ["姓名", "民族", "性别", "出生", "住址", "签发机关", "有效期限", "公民身份号码"]

    new_data = []
    i = 0
    while i < len(data):
        item = data[i]
        prev_item = ''
        for keyword in keywords:
            if keyword in item:
                # 分割关键字前后的内容
                split_index = item.find(keyword)
                left_part = item[:split_index].strip()

                # 处理粘连的关键字
                while keyword in item[split_index + len(keyword):]:
                    next_split_index = item[split_index + len(keyword):].find(keyword)
                    next_split_index += split_index + len(keyword)
                    right_part = item[split_index + len(keyword):next_split_index]
                    new_data.append(f"{prev_item}{right_part.strip()}")

                    item = item[next_split_index:]
                    split_index = 0

                right_part = item[split_index + len(keyword):].strip()
                new_data.append(f"{keyword}{left_part}")
                new_data.append(f"{keyword}{right_part}")
                i += 1
                break
        else:
            # 非关键字项或标准格式的项，原样添加
            new_data.append(item)
            i += 1

    return new_data


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=12306)
