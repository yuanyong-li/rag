# RAG复杂问题处理链路主流程
import json
import logging
from typing import List, Dict, Any

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from .prompt import (
    decompose_query_prompt,
    subquestion_answer_prompt,
    check_relevance_prompt,
    final_answer_prompt,
)
from .tools import (
    chat_completions4, doc_2_doclist,
    setup_logger, query_faults
)

logger = setup_logger("MyLogger", logging.DEBUG)


class SubQuestionAnswer(BaseModel):
    reference: str = Field(description="用于推理的文档原文")
    answer: str = Field(description="基于引用内容进行推理的答案")


class RelevanceCheck(BaseModel):
    relevance: bool = Field(description="回答是否解答了子问题")


class Hallucination_Check(BaseModel):
    relevance: bool = Field(description="回答是否与子问题相关")
    faithfulness: bool = Field(description="回答是否严格基于证据")
    evidence_from_document: bool = Field(description="证据是否严格来自源参考文档")


class FinalAnswer(BaseModel):
    answer: str = Field(description="最终答案")


def generate_subquestions(complex_query, scene_tag, province_tag="hq"):
    """
    输入复杂Query，生成m个子问题，返回结构化JSON数组。
    """
    # 创建输出解析器
    parser = JsonOutputParser()

    # 构造 prompt
    prompt = PromptTemplate(
        template=decompose_query_prompt,
        input_variables=["complex_query"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()}
    )

    # 调用大模型
    response = chat_completions4(prompt.format(complex_query=complex_query))
    if hasattr(response, "choices"):
        content = response.choices[0].message.content
    else:
        content = str(response)

    # 使用解析器解析输出
    try:
        subquestions = parser.parse(content)
        assert isinstance(subquestions, list)
        return subquestions
    except Exception as e:
        raise ValueError(f"模型输出解析失败，内容为：{content}\n错误信息：{e}")


def retrieve_docs_for_subquestions(subquestions, scene_tag, province_tag="hq", k=5):
    """
    针对每个子问题检索k个文档。
    """
    subquestions_docs = []
    for query_text in subquestions:
        rag_result = query_faults(query_text, scene_tag, province_tag="hq",
                                  top_k=5, score_threshold=0.5)
        if len(rag_result["data"]["context"]) == 0:
            rag_result = query_faults(query_text, ["wlyh", "wxwy", "xczc", "yyjc", "xczhw"], province_tag="hq",
                                      top_k=5, score_threshold=0.5)
        subquestions_docs.append((query_text, rag_result["data"]["context"]))
    return subquestions_docs


def answer_subquestions_with_llm(subquestions_with_docs):
    """
    LLM逐个回答子问题，基于检索到的文档生成结构化答案。

    该函数接收子问题和对应的文档列表，使用大语言模型为每个子问题生成答案。
    支持处理多个文档，并自动合并文档内容用于生成答案。

    Args:
        subquestions_with_docs (list): 子问题和文档的配对列表
            每个元素为元组: (subquestion, docs)
            - subquestion (str): 子问题文本
            - docs (list): 文档列表，每个文档为字典格式
                - doc_name (str): 文档名称
                - img_url (str): 图片URL（可选）
                - text (str): 文档文本内容
                - url (str): 文档URL（可选）

    Returns:
        list: 子问题回答结果列表
            每个元素为元组: (subquestion, docs, subanswer)
            - subquestion (str): 原始子问题
            - docs (list): 原始文档列表
            - subanswer (dict): 结构化回答
                - reference (str): 引用的文档原文片段
                - answer (str): 基于文档的推理答案

    Example:
        >>> subquestions_with_docs = [
        ...     ("什么是人工智能？", [
        ...         {"text": "人工智能是计算机科学的一个分支...", "doc_name": "AI介绍"}
        ...     ]),
        ...     ("AI有哪些应用？", [
        ...         {"text": "机器学习、自然语言处理...", "doc_name": "AI应用"}
        ...     ])
        ... ]
        >>> results = answer_subquestions_with_llm(subquestions_with_docs)
        >>> # 返回: [("什么是人工智能？", docs, {"reference": "...", "answer": "..."}), ...]

    Note:
        - 如果文档列表为空，会使用"无相关文档"作为输入
        - 多个文档会自动编号并合并文本内容
        - 解析失败时会返回空的answer字段
    """
    parser = JsonOutputParser(pydantic_object=SubQuestionAnswer)

    # 创建prompt模板
    prompt = PromptTemplate(
        template=subquestion_answer_prompt,
        input_variables=["question", "document"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()}
    )

    results = []
    print(subquestions_with_docs)
    for subq, docs in subquestions_with_docs:
        # 处理文档内容
        doc_texts = doc_2_doclist(docs)
        document_content = "\n\n".join(doc_texts) if doc_texts else "无相关文档"

        # 生成回答
        formatted_prompt = prompt.format(
            question=subq, document=document_content)
        response = chat_completions4(formatted_prompt)

        try:
            if hasattr(response, "choices"):
                content = response.choices[0].message.content
            else:
                content = str(response)

            result = parser.parse(content)
            subanswer = result
        except Exception as e:
            # 如果解析失败，提供默认值
            subanswer = {
                "reference": "",
                "answer": ""
            }

        # 返回三元组：(subquestion, docs, subanswer)
        results.append((subq, docs, subanswer))

    return results


def check_faithfulness_and_relevance(answers_with_docs):
    parser = JsonOutputParser(pydantic_object=Hallucination_Check)

    # 创建prompt模板
    prompt = PromptTemplate(
        template=check_relevance_prompt,
        input_variables=["subquestion", "answer", "document"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()}
    )

    results = []
    for subq, docs, subanswer in answers_with_docs:
        # 提取文档文本内容用于faithfulness检查
        doc_texts = doc_2_doclist(docs)

        # faithfulness: 判断答案是否引用了文档内容
        answer_text = subanswer.get("answer", "")
        evidence = subanswer.get("reference", "")

        # relevance: 由LLM判断
        formatted_prompt = prompt.format(
            subquestion=subq, answer=answer_text, evidence=evidence, document=doc_texts)
        response = chat_completions4(formatted_prompt)

        try:
            if hasattr(response, "choices"):
                content = response.choices[0].message.content
            else:
                content = str(response)

            result = parser.parse(content)
            relevance = result["relevance"]
            faithfulness = result["faithfulness"]
            evidence_from_document = result["evidence_from_document"]
        except Exception as e:
            relevance = False
            faithfulness = False
            evidence_from_document = False
            print(f"check_faithfulness_and_relevance 返回结果解析异常")

        results.append({
            "subquestion": subq,
            "docs_per_subq": docs,
            "answer": answer_text,
            "evidence": evidence,
            "faithfulness": faithfulness,
            "relevance": relevance,
            "evidence_from_document": evidence_from_document,
        })
    return results


def build_structured_evidence(subquestions_docs_subanswer):
    """
    拼接子问题+答案或子问题+参考文档构成structured evidence。
    faithfulness和relevance都为true时，拼接子问题-子答案；否则拼接子问题-参考文档。
    返回所有数据对组成的列表。
    """
    structured_evidence = []
    for item in subquestions_docs_subanswer:
        subq = item["subquestion"]
        docs = item.get("docs_per_subq", [])
        answer = item.get("answer", "")
        faithfulness = item.get("faithfulness", False)
        relevance = item.get("relevance", False)
        evidence_from_document = item.get("evidence_from_document", False)

        docs_texts = doc_2_doclist(docs)

        if not faithfulness or not relevance or not evidence_from_document:
            structured_evidence.append({
                "subquestion": subq,
                "evidence": docs_texts
            })
        else:
            structured_evidence.append({
                "subquestion": subq,
                "evidence": answer
            })
    return structured_evidence


def build_structured_evidence_baseline(subquestions_docs):
    """
    拼接子问题+答案或子问题+参考文档构成structured evidence。
    faithfulness和relevance都为true时，拼接子问题-子答案；否则拼接子问题-参考文档。
    返回所有数据对组成的列表。
    """
    structured_evidence = []
    for item in subquestions_docs:
        subq = item[0]
        docs = item[1]
        docs_texts = doc_2_doclist(docs)

        structured_evidence.append({
            "subquestion": subq,
            "evidence": docs_texts
        })
    return structured_evidence


def final_answer_with_rag_fusion(complex_query, structured_evidence):
    """
    LLM生成最终复杂问题答案（用RAG-Fusion做路径融合）。
    要求模型严格按照提供的子问题答案或文档回答，不要编造内容。
    """
    # 创建输出解析器
    parser = JsonOutputParser(pydantic_object=FinalAnswer)

    # 创建prompt模板
    prompt = PromptTemplate(
        template=final_answer_prompt,
        input_variables=["complex_query", "doc_anwer"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()}
    )

    # 构造结构化证据字符串
    evidence_str = ""
    for i, item in enumerate(structured_evidence, 1):
        evidence_str += f"子问题{i}: {item['subquestion']}\n"
        evidence_str += f"{item['evidence']}\n\n"

    formatted_prompt = prompt.format(
        complex_query=complex_query,
        structured_evidence=evidence_str
    )

    response = chat_completions4(formatted_prompt)

    try:
        if hasattr(response, "choices"):
            content = response.choices[0].message.content
        else:
            content = str(response)

        result = parser.parse(content)
        return result, formatted_prompt
    except Exception as e:
        return {
            "answer": f"final_answer_with_rag_fusion解析失败，content：{content}",
        }, formatted_prompt


def main():
    complex_query = "武汉5G高回落小区的主要原因是什么，它们的数量下降了多少，是否有可能通过调整某些技术手段来进一步减少回落现象？"
    scene_tag = 'wlyh'
    k = 5
    subquestions = generate_subquestions(complex_query, scene_tag)
    subquestions__docs = retrieve_docs_for_subquestions(
        subquestions, scene_tag, k)
    subquestions_docs_subanswer = answer_subquestions_with_llm(
        subquestions__docs)
    checked_subquestions_docs_subanswer = check_faithfulness_and_relevance(
        subquestions_docs_subanswer)
    structured_evidence = build_structured_evidence(
        checked_subquestions_docs_subanswer)
    final_answer, _ = final_answer_with_rag_fusion(
        complex_query, structured_evidence)

    logger.info(f"复杂问题: {complex_query}")
    logger.info(f"最终答案: {final_answer['answer']}")


if __name__ == "__main__":
    main()
