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

        # 3. Tạo Context (CÓ SỬA ĐỔI QUAN TRỌNG)
        # Thay vì chỉ join text, ta sẽ lấy cả link ảnh từ metadata ghép vào
        context_parts = []
        source_names = [] # Lưu tên món để debug

        for n in rerank_nodes:
            # Lấy nội dung văn bản
            text_content = n.node.text
            
            # Lấy link ảnh từ metadata
            image_link = n.node.metadata.get("image_link", "")
            
            # Kiểm tra nếu có link ảnh hợp lệ thì ghép vào text
            # Format này giúp LLM hiểu: "À, đoạn văn này có cái ảnh đi kèm ở đây"
            if image_link and str(image_link) != "nan" and str(image_link).startswith("http"):
                text_content += f"\n[IMAGE_URL: {image_link}]"
            
            context_parts.append(text_content)
            
            # Lấy tên món ăn để trả về metadata (giúp API hiển thị source)
            dish_name = n.node.metadata.get("dish_name", "Unknown")
            source_names.append(dish_name)

        # Nối các đoạn lại với nhau
        context = "\n---------------------\n".join(context_parts)

        # ==========================================================
        # 4. GẮN PROMPT LUCFIN
        # ==========================================================
        formatted_prompt = qa_prompt_tmpl.format(
            context_str=context, 
            query_str=query_bundle.query_str
        )

        # 5. Gửi cho LLM trả lời
        answer = self.llm.complete(formatted_prompt)

        # 6. Trả về Response kèm Source Nodes (quan trọng để API lấy được metadata gốc)
        # Lưu ý: LlamaIndex cần source_nodes để truy xuất metadata sau này
        return Response(
            response=str(answer), 
            source_nodes=rerank_nodes, # <-- TRẢ VỀ CẢ NODE GỐC
            metadata={"doc_ids": source_names}
        )

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        raise NotImplementedError("Async query not supported.")
        
    def _get_prompt_modules(self) -> dict:
        return {}