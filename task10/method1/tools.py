import logging
import requests
import json
from openai import OpenAI


def query_faults(query_text, scene_tag, province_tag="hq", top_k=5, score_threshold=0.5):
    """
    调用RAG
    """
    # url = "http://10.141.179.170:20028/bm/query/kg/trace"
    # url = "http://10.128.86.64:8000/serviceAgent/rest/bm/query/kg/new/trag"
    url = "http://10.141.179.170:20028/bm/query/kg/trag"
    headers = {
        "Content-Type": "application/json",
        "X-APP-ID": "91d71bebe01b563ff5a5add03c27ca54",
        "X-APP-KEY": "ea15864c66c22c4f0f88e47a8d1cf0d5"
    }

    # 处理scene_tag为列表的情况
    if isinstance(scene_tag, list):
        # 为每个scene_tag创建一个字典
        tag_list = []
        for tag in scene_tag:
            tag_dict = {
                "scene_tag": tag,
                "province_tag": province_tag,
                # "specialty_tag": "jrw_zhw"
            }
            tag_list.append(tag_dict)
    else:
        # scene_tag为字符串的情况
        tag_list = [
            {
                "scene_tag": scene_tag,
                "province_tag": province_tag,
                # "specialty_tag": "jrw_zhw"
            }
        ]

    body = {
        "query": query_text,
        "tag": tag_list,
        "top_k": top_k,
        "show_image": True,
        "score_threshold": score_threshold
    }

    response = requests.post(url, headers=headers, data=json.dumps(body))
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "status_code": response.status_code, "text": response.text}


def chat_completions4(query):
    API_SECRET_KEY = "sk-zk25ec58acf44398bd0c18450792ecc97216808f473e2113"
    BASE_URL = "https://api.zhizengzeng.com/v1/"
    client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
    resp = client.chat.completions.create(
        model="qwen3-32b",
        messages=[
            {"role": "user", "content": query}
        ]
    )
    return resp


def doc_2_doclist(docs):
    if docs and len(docs) > 0:
        # 提取所有文档的text内容
        doc_texts = []
        for idx, doc in enumerate(docs, 1):  # 使用 enumerate 给每个文档编号
            if 'text' in doc and doc['text']:
                # 为每个文档添加编号或说明
                doc_texts.append(
                    f"文档 {idx} 内容:\n{doc['text']}\n")  # 文档编号和内容
        return doc_texts
    else:
        return ["无相关文档"]


def setup_logger(name, log_level=logging.INFO):
    """
    设置日志记录器

    参数:
    - name: 日志记录器名称
    - log_level: 日志级别，默认为 INFO
    """
    # 创建一个logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # 创建一个控制台输出Handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    # 创建日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
    ch.setFormatter(formatter)

    # 给logger添加Handler
    logger.addHandler(ch)

    return logger
