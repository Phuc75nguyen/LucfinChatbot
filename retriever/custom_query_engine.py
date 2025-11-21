from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.schema import QueryBundle
from retriever.custom_retriever import ChromaDBRetriever
from retriever.query_handling import rerank_and_normalize

class NutritionQueryEngine(BaseQueryEngine):
    def __init__(self, retriever: ChromaDBRetriever, llm, reranker = None, callback_manager=None):
        super().__init__(callback_manager=callback_manager)
        self.retriever = retriever
        self.llm = llm
        self.reranker = reranker

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        nodes = self.retriever._retrieve(query_bundle)

        rerank_nodes = self.reranker.postprocess_nodes(nodes, query_bundle=query_bundle)
        
        if len(rerank_nodes) == 0:
            return Response(response="RAG: " + "Không tìm thấy thông tin nào liên quan trong toàn bộ tài liệu.",
                            metadata={"doc_ids": None})

        context = "\n".join([n.node.text for n in rerank_nodes])
        prompt = f"Trả lời câu hỏi không sử dụng thông tin đã được train từ trước chỉ dựa trên thông tin sau:\n\n{context}\n\nCâu hỏi: {query_bundle.query_str}"
        answer = self.llm.complete(prompt)

        doc_ids = list(set([
            n.node.metadata.get("doc_id")
            for n in rerank_nodes
            if n.node.metadata.get("doc_id") is not None
        ])) or None

        return Response(response=answer, metadata={"doc_ids": doc_ids})

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        raise NotImplementedError("Async query not supported.")
    def _get_prompt_modules(self) -> dict:
        return {}

