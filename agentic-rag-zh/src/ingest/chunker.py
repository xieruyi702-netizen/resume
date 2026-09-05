"""父子分块：child 短块用于检索精度，parent 长块用于上下文完整性。

中文分块用中文标点做切分边界；news 语料单篇约 600 字，
大多数文档 parent = 全文、children = 2~3 个 256 字短块。
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config

_SEPARATORS = ["\n\n", "\n", "。", "；", "，", " "]


def split_doc(doc_id: str, text: str) -> tuple[list[dict], list[dict]]:
    """一篇文档 → (parents, children)。

    parents: [{parent_id, doc_id, text}]
    children: [{child_id, parent_id, doc_id, text}]
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.PARENT_CHUNK, chunk_overlap=0, separators=_SEPARATORS
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK, chunk_overlap=config.CHILD_OVERLAP, separators=_SEPARATORS
    )
    parents, children = [], []
    for pi, ptext in enumerate(parent_splitter.split_text(text)):
        parent_id = f"{doc_id}#p{pi}"
        parents.append({"parent_id": parent_id, "doc_id": doc_id, "text": ptext})
        for ci, ctext in enumerate(child_splitter.split_text(ptext)):
            children.append({
                "child_id": f"{parent_id}#c{ci}",
                "parent_id": parent_id,
                "doc_id": doc_id,
                "text": ctext,
            })
    return parents, children
