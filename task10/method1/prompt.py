decompose_query_prompt = """
你是一名专业的知识工程师，擅长将复杂问题拆解为若干可独立检索和回答的子问题。

请将以下复杂问题拆解为可以独立检索和回答的子问题，并以 **JSON 数组格式** 输出，例如：
[
  "子问题1",
  "子问题2",
  "子问题3"
]

要求：
- 子问题应覆盖原问题的所有核心要素。
- 子问题应简明、具体、无歧义。
- 子问题之间应无重复、尽量无交叉。
- 不要输出解释或额外文字，仅输出最终的 JSON。

{format_instructions}

复杂问题：
{complex_query}
"""

subquestion_answer_prompt = """
你是一名严谨的研究助理，必须**只根据提供的参考文档**来回答用户的问题。

现在请你阅读以下参考资料，并回答给定子问题。

要求：
- 回答必须**基于下方提供的文档内容 **，不允许引入外部知识。
- 输出格式必须为 JSON，包含以下字段：
- "reference"：你用于推理的文档原文（可精简片段，但必须是真实引用）
- "answer"：基于引用内容进行推理的答案
- 不要输出任何解释、说明或额外内容，只输出合法的 JSON。

{format_instructions}

以下是问题与参考文档：

问题：
{question}

参考文档：
{document}
"""

check_relevance_prompt = """
你是一名严谨的学术评审专家。请判断下面的“回答”是否满足以下三个标准，并只输出 JSON 格式的结论：

1. **相关性**：回答是否直接回答了“子问题”。
2. **可信度**：回答是否严格基于提供的证据。
3. **证据来源**：证据是否严格来自于提供的文档内容。

要求：
- 只根据子问题、回答、回答证据和源参考文档进行判断，不引入外部知识。
- 输出 JSON 格式，包含以下字段：
  - "relevance"：判断回答是否与子问题相关，值为 `true` 或 `false`。
  - "faithfulness"：判断回答是否严格基于回答证据，值为 `true` 或 `false`，evidence为空则为`true`。
  - "evidence_from_document"：判断证据是否严格来自源参考文档，值为 `true` 或 `false`，document为空则为`true`。

{format_instructions}

子问题：
{subquestion}

回答：
{answer}

回答证据：
{evidence}

源参考文档：
{document}
"""


final_answer_prompt = """
你是一名严谨的研究专家。请根据以下结构化证据回答复杂问题。

要求：
- 必须严格按照提供的子问题答案或文档内容进行回答
- 不允许引入任何外部知识或编造内容
- 如果因为证据不足无法回答，请输出无法回答
- 输出格式必须为JSON，包含以下字段：
  - "answer": "基于证据的综合回答"

{format_instructions}

复杂问题：
{complex_query}

[子问题-答案]或[子问题-参考文档]：
{structured_evidence}

"""
