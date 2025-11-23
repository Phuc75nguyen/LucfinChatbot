from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.schema import QueryBundle
from retriever.custom_retriever import ChromaDBRetriever
from retriever.query_handling import rerank_and_normalize

# 1. Import cái prompt xịn vào
from router.prompt import qa_prompt_tmpl 

class NutritionQueryEngine(BaseQueryEngine):
    def __init__(self, retriever: ChromaDBRetriever, llm, reranker=None, callback_manager=None):
        super().__init__(callback_manager=callback_manager)
        self.retriever = retriever
        self.llm = llm
        self.reranker = reranker

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        # 1. Truy xuất dữ liệu (Retrieve)
        nodes = self.retriever._retrieve(query_bundle)

        # 2. Xếp hạng lại (Rerank)
        rerank_nodes = self.reranker.postprocess_nodes(nodes, query_bundle=query_bundle)
        
        # Xử lý trường hợp không tìm thấy
        if len(rerank_nodes) == 0:
            return Response(
                response="RAG: " + "Xin lỗi, hiện tại Lucfin chưa có dữ liệu về món ăn này trong hệ thống.",
                metadata={"doc_ids": None}
            )

        # 3. Tạo Context (Ghép các đoạn văn bản lại)
        context = "\n".join([n.node.text for n in rerank_nodes])

        # ==========================================================
        # 4.  GẮN PROMPT LUCFIN VÀO ĐÂY
        # ==========================================================
        # Template sẽ tự điền context vào chỗ {context_str} và câu hỏi vào chỗ {query_str}
        formatted_prompt = qa_prompt_tmpl.format(
            context_str=context, 
            query_str=query_bundle.query_str
        )

        # 5. Gửi cho LLM trả lời
        # Lúc này LLM sẽ nhận được toàn bộ hướng dẫn "Bạn là chuyên gia Lucfin..."
        answer = self.llm.complete(formatted_prompt) # Lưu ý: Dùng formatted_prompt

        # 6. Trích xuất Doc IDs để debug
        doc_ids = list(set([
            n.node.metadata.get("doc_id")
            for n in rerank_nodes
            if n.node.metadata.get("doc_id") is not None
        ])) or None

        return Response(response=str(answer), metadata={"doc_ids": doc_ids})

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        raise NotImplementedError("Async query not supported.")
        
    def _get_prompt_modules(self) -> dict:
        return {}