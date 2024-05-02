from llama_index.core import PromptTemplate

context_str =""""""

qa_prompt_tmpl_str = """\
    Context information is below.
    ---------------------

    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {query_str}
    Answer: \
"""

# qa_prompt_tmpl_str = """\
#     Context information is below.
#     ---------------------
#     {context_str}
#     ---------------------
#     Given the context information and not prior knowledge, answer the query.
#     Query: {query_str}
#     Answer: \
# """

