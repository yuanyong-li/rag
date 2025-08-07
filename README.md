# RAG复杂问题处理系统

## 项目简介

本项目是一个基于检索增强生成（RAG）的复杂问题处理系统，专门用于处理多步骤、多子问题的复杂查询。系统通过问题分解、文档检索、子问题回答和答案融合等步骤，实现对复杂问题的准确回答。

## 功能特性

### 🔍 核心功能
- **复杂问题分解**：将复杂查询自动分解为多个子问题
- **智能文档检索**：基于语义相似度检索相关文档
- **子问题回答**：使用大语言模型基于检索文档回答子问题
- **答案质量检查**：检查回答的相关性和忠实度
- **答案融合**：将子问题答案融合为最终答案

### 📊 支持场景
- **网络运维问题**：5G网络优化、基站验收等
- **技术故障分析**：网络故障诊断、性能优化等
- **多维度查询**：涉及多个方面的复杂技术问题

## 项目结构

```
rag/
├── README.md                 # 项目说明文档
├── outputs/                  # 输出结果目录
│   ├── MOS质差_baseline.json
│   ├── MOS质差_method1.json
│   ├── 基站远程验_baseline.json
│   └── 基站远程验_method1.json
└── task10/                   # 主要代码目录
    ├── baseline_test.ipynb   # 基线方法测试
    ├── method1_test.ipynb    # 改进方法测试
    ├── rag_request.ipynb     # RAG请求示例
    ├── test_function.py      # 测试函数
    └── method1/              # 改进方法实现
        ├── main.py           # 主流程实现
        ├── tools.py          # 工具函数
        ├── prompt.py         # 提示词模板
        ├── config.py         # 配置文件
        ├── requirements.txt  # 依赖包列表
        └── __init__.py
```

## 方法对比

### Baseline方法
- **流程**：问题分解 → 文档检索 → 直接融合
- **特点**：简单直接，处理速度快
- **适用**：简单复杂问题

### Method1（改进方法）
- **流程**：问题分解 → 文档检索 → 子问题回答 → 质量检查 → 答案融合
- **特点**：增加质量检查，提高答案准确性
- **适用**：高质量要求的复杂问题

## 安装与配置

### 环境要求
- Python 3.8+
- OpenAI API密钥

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd rag
```

2. **安装依赖**
```bash
cd task10/method1
pip install -r requirements.txt
```

3. **配置API密钥**
```bash
# 复制配置文件
cp config_example.py config.py

# 编辑config.py，添加您的API密钥
OPENAI_API_KEY = "your-api-key-here"
```

### 依赖包列表

```
# 核心依赖
langchain-core>=0.1.0
langchain-openai>=0.1.0
openai>=1.0.0
pydantic>=1.10.0
typing-extensions>=4.0.0

# HTTP请求
requests>=2.25.0

# 数据处理
json5>=0.9.0

# 日志和工具
python-dotenv>=0.19.0

# 可选：用于更好的JSON处理
jsonschema>=3.2.0
```

## 使用方法

### 快速开始

1. **运行基线方法测试**
```python
from test_function import baseline_test

# 测试复杂问题
complex_query = "武汉5G高回落小区的主要原因是什么，它们的数量下降了多少，是否有可能通过调整某些技术手段来进一步减少回落现象？"
result = baseline_test(complex_query, 'wlyh', 5)
```

2. **运行改进方法测试**
```python
from test_function import method1_test

# 测试复杂问题
complex_query = "基站远程验收系统是如何提升验收效率和入网质量的？"
result = method1_test(complex_query, 'wlyh_wxwy', 5)
```

### 核心API

#### 问题分解
```python
from method1.main import generate_subquestions

subquestions = generate_subquestions(complex_query, scene_tag)
```

#### 文档检索
```python
from method1.main import retrieve_docs_for_subquestions

subquestions_docs = retrieve_docs_for_subquestions(subquestions, scene_tag)
```

#### 子问题回答
```python
from method1.main import answer_subquestions_with_llm

answers = answer_subquestions_with_llm(subquestions_with_docs)
```

#### 质量检查
```python
from method1.main import check_faithfulness_and_relevance

checked_answers = check_faithfulness_and_relevance(answers_with_docs)
```

## 性能对比

### 处理时间对比
| 方法 | 问题分解 | 文档检索 | 子问题回答 | 质量检查 | 答案融合 | 总时间 |
|------|----------|----------|------------|----------|----------|--------|
| Baseline | 6.9s | 7.0s | - | - | 18.0s | 31.9s |
| Method1 | 4.6s | 6.7s | 30.3s | 10.6s | 11.5s | 63.7s |

### 答案质量
- **Baseline**：直接基于检索文档生成答案，速度快但可能缺乏深度分析
- **Method1**：通过子问题回答和质量检查，提供更准确、更可靠的答案

## 示例输出

### 输入问题
```
武汉5G高回落小区的主要原因是什么，它们的数量下降了多少，是否有可能通过调整某些技术手段来进一步减少回落现象？
```

### Method1输出
```json
{
  "answer": "武汉5G高回落小区的主要原因是覆盖类问题，占比87.15%，包括深度覆盖不足、孤岛、弱覆盖、切换邻区等问题。武汉5G高回落小区的数量下降了163个。可以通过增加幅度磁滞hys和增加time-to-trigger的值来减少5G辅站间的来回频繁变更，从而进一步减少回落现象。"
}
```

## 文档数据

项目支持多种场景的文档检索：
- **wlyh**：网络运维相关文档
- **wlyh_wxwy**：网络运维与无线网络文档
- **xczc**：现场支撑文档
- **yyjc**：语音检测文档
- **xczhw**：现场支撑与无线网络文档

## 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件至项目维护者

---

**注意**：使用前请确保已正确配置OpenAI API密钥，并遵守相关使用条款。

