# -*- coding: utf-8 -*-
# @File Name: 识图.py
# @Created Time: 2024/3/4 12:03
# @SoftWare: PyCharm
from paddleocr import PaddleOCR
import os

def extract_tuple_first_element(input_list):
    result = []
    for x in input_list:
        if isinstance(x, tuple):
            result.append(x[0])
        elif isinstance(x, list):
            result.extend(extract_tuple_first_element(x))
    return result


def ocr_PaddleOCR(img):
    result_str = "✅识别完成，结果如下：\n"
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False, show_log=False)
    result = ocr.ocr(img, cls=True)
    result = extract_tuple_first_element(result)
    if len(result) == 0:
        return "❌未识别到内容，请换一张图片重试"
    for i in range(len(result)):
        result_str += f"{result[i]}\n"
    return result_str

image_path = os.path.join(os.path.dirname(__file__), "images", "a.jpg")
print(ocr_PaddleOCR(image_path))
