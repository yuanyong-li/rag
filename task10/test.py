from method1.tools import chat_completions4, doc_2_doclist, setup_logger
import method1.main as main
import logging


def baseline_test(complex_query, scene_tag, k):
    logger = setup_logger("MyLogger", logging.DEBUG)
    complex_query = "武汉5G高回落小区的主要原因是什么，它们的数量下降了多少，是否有可能通过调整某些技术手段来进一步减少回落现象？"
    scene_tag = 'wlyh'
    k = 5
    subquestions = main.generate_subquestions(complex_query, scene_tag)
    subquestions_docs = main.retrieve_docs_for_subquestions(
        subquestions, scene_tag, k)
    structured_evidence = main.build_structured_evidence_baseline(
        subquestions_docs)
    final_answer, final_prompt = main.final_answer_with_rag_fusion(
        complex_query, structured_evidence)

    logger.info(f"复杂问题: {complex_query}")
    logger.info(f"最终答案: {final_answer['answer']}")

    return subquestions, subquestions_docs, structured_evidence, \
        final_answer, final_prompt


def method1_test(complex_query, scene_tag, k):
    logger = setup_logger("MyLogger", logging.DEBUG)
    subquestions = main.generate_subquestions(complex_query, scene_tag)
    subquestions_docs = main.retrieve_docs_for_subquestions(
        subquestions, scene_tag, k)
    subquestions_docs_subanswer = main.answer_subquestions_with_llm(
        subquestions_docs)
    checked_subquestions_docs_subanswer = main.check_faithfulness_and_relevance(
        subquestions_docs_subanswer)
    structured_evidence = main.build_structured_evidence(
        checked_subquestions_docs_subanswer)
    final_answer, final_prompt = main.final_answer_with_rag_fusion(
        complex_query, structured_evidence)

    logger.info(f"复杂问题: {complex_query}")
    logger.info(f"最终答案: {final_answer}")
    return subquestions, subquestions_docs, \
        subquestions_docs_subanswer, checked_subquestions_docs_subanswer, \
        structured_evidence, final_answer, final_prompt


if __name__ == "__main__":
    baseline_test(
        "武汉5G高回落小区的主要原因是什么，它们的数量下降了多少，是否有可能通过调整某些技术手段来进一步减少回落现象？", 'wlyh', 5)
