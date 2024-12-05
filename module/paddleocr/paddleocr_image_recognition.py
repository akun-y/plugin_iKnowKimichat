# coding=utf-8
"""
Author: chazzjimel
Email: chazzjimel@gmail.com
wechat：cheung-z-x

Description:

"""
from loguru import logger
import requests

#from common.log import logger

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
        return -1, "❌未识别到内容，请换一张图片重试"
    for i in range(len(result)):
        result_str += f"{result[i]}\n"
    return 0, result_str,result

def analyze_image_paddle(file_path):
   
    try:        
        err_code, ocr_result,ocr_result_list = ocr_PaddleOCR(file_path)
        logger.info(f"========>图片OCR结果：{ocr_result}")        

        return {
            "result": ocr_result,
            "result_list": ocr_result_list,
            "type": "paddleocr",
            "code": err_code
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"PADDLEOCR图像分析出错: {e}")
        return None
    except IOError as e:
        logger.error(f"PADDLEOCR文件操作出错: {e}")
        return None
