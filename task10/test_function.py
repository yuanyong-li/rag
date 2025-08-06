from method1.tools import chat_completions4, doc_2_doclist, setup_logger
import method1.main as main
import logging
import time


def baseline_test(complex_query, scene_tag, k):
    logger = setup_logger("MyLogger", logging.DEBUG)
    complex_query = "武汉5G高回落小区的主要原因是什么，它们的数量下降了多少，是否有可能通过调整某些技术手段来进一步减少回落现象？"
    scene_tag = 'wlyh'
    k = 5
    
    # 用于存储时间统计的列表
    time_stats = []
    
    # 生成子问题
    start_time = time.time()
    subquestions = main.generate_subquestions(complex_query, scene_tag)
    end_time = time.time()
    time_stats.append(("generate_subquestions", end_time - start_time))
    
    # 检索文档
    start_time = time.time()
    subquestions_docs = main.retrieve_docs_for_subquestions(
        subquestions, scene_tag, k)
    end_time = time.time()
    time_stats.append(("retrieve_docs_for_subquestions", end_time - start_time))
    
    # 构建结构化证据
    start_time = time.time()
    structured_evidence = main.build_structured_evidence_baseline(
        subquestions_docs)
    end_time = time.time()
    time_stats.append(("build_structured_evidence_baseline", end_time - start_time))
    
    # 最终答案生成
    start_time = time.time()
    final_answer, final_prompt = main.final_answer_with_rag_fusion(
        complex_query, structured_evidence)
    end_time = time.time()
    time_stats.append(("final_answer_with_rag_fusion", end_time - start_time))

    logger.info(f"复杂问题: {complex_query}")
    logger.info(f"最终答案: {final_answer['answer']}")
    
    # 打印时间统计
    logger.info("时间统计:")
    for func_name, duration in time_stats:
        logger.info(f"  {func_name}: {duration:.4f}秒")

    return subquestions, subquestions_docs, structured_evidence, \
        final_answer, final_prompt, time_stats


def method1_test(complex_query, scene_tag, k):
    logger = setup_logger("MyLogger", logging.DEBUG)
    
    # 用于存储时间统计的列表
    time_stats = []
    
    # 生成子问题
    start_time = time.time()
    subquestions = main.generate_subquestions(complex_query, scene_tag)
    end_time = time.time()
    time_stats.append(("generate_subquestions", end_time - start_time))
    
    # 检索文档
    start_time = time.time()
    subquestions_docs = main.retrieve_docs_for_subquestions(
        subquestions, scene_tag, k)
    end_time = time.time()
    time_stats.append(("retrieve_docs_for_subquestions", end_time - start_time))
    
    # 使用LLM回答子问题
    start_time = time.time()
    subquestions_docs_subanswer = main.answer_subquestions_with_llm(
        subquestions_docs)
    end_time = time.time()
    time_stats.append(("answer_subquestions_with_llm", end_time - start_time))
    
    # 检查忠实性和相关性
    start_time = time.time()
    checked_subquestions_docs_subanswer = main.check_faithfulness_and_relevance(
        subquestions_docs_subanswer)
    end_time = time.time()
    time_stats.append(("check_faithfulness_and_relevance", end_time - start_time))
    
    # 构建结构化证据
    start_time = time.time()
    structured_evidence = main.build_structured_evidence(
        checked_subquestions_docs_subanswer)
    end_time = time.time()
    time_stats.append(("build_structured_evidence", end_time - start_time))
    
    # 最终答案生成
    start_time = time.time()
    final_answer, final_prompt = main.final_answer_with_rag_fusion(
        complex_query, structured_evidence)
    end_time = time.time()
    time_stats.append(("final_answer_with_rag_fusion", end_time - start_time))

    logger.info(f"复杂问题: {complex_query}")
    logger.info(f"最终答案: {final_answer}")
    
    # 打印时间统计
    logger.info("时间统计:")
    for func_name, duration in time_stats:
        logger.info(f"  {func_name}: {duration:.4f}秒")
    
    return subquestions, subquestions_docs, \
        subquestions_docs_subanswer, checked_subquestions_docs_subanswer, \
        structured_evidence, final_answer, final_prompt, time_stats


if __name__ == "__main__":
    baseline_test(
        "武汉5G高回落小区的主要原因是什么，它们的数量下降了多少，是否有可能通过调整某些技术手段来进一步减少回落现象？", 'wlyh', 5)
