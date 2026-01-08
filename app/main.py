from fastapi import FastAPI, HTTPException
from typing import List, Optional
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from io import BytesIO
import os

app = FastAPI(title="Tour Recommendation AI Service")

# ===============================
# LLM Configuration (GGUF)
# ===============================
GGUF_MODEL_PATH = os.getenv(
    "GGUF_MODEL_PATH", "/usr/src/app/model/qwen2.5-1.5b-tour-assistant-q4.gguf"
)
N_CTX = int(os.getenv("N_CTX", "2048"))  # Context window
N_THREADS = int(os.getenv("N_THREADS", "4"))  # CPU threads
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "0"))  # GPU layers (0 = CPU only)

llm = None

# ===============================
# Embedding Models
# ===============================
# Multilingual MiniLM for text
text_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# CLIP for images
image_model = SentenceTransformer("clip-ViT-B-32")


# ===============================
# Request/Response Models
# ===============================
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7


class FusedVectorRequest(BaseModel):
    title: str
    description: str
    summary: str
    address: str
    image_urls: List[str]
    price: float
    duration_days: float
    min_price_global: float = 0.0
    max_price_global: float = 100000000.0
    max_duration_global: float = 30.0


# ===============================
# LLM Loading (Lazy)
# ===============================
def load_llm():
    """Load GGUF model using llama-cpp-python"""
    global llm
    if llm is None:
        try:
            from llama_cpp import Llama

            print(f"Loading GGUF model: {GGUF_MODEL_PATH}")
            print(
                f"Config: n_ctx={N_CTX}, n_threads={N_THREADS}, n_gpu_layers={N_GPU_LAYERS}"
            )

            if not os.path.exists(GGUF_MODEL_PATH):
                print(f"WARNING: Model file not found at {GGUF_MODEL_PATH}")
                llm = "NOT_FOUND"
                return

            llm = Llama(
                model_path=GGUF_MODEL_PATH,
                n_ctx=N_CTX,
                n_threads=N_THREADS,
                n_gpu_layers=N_GPU_LAYERS,
                verbose=False,
            )
            print("GGUF model loaded successfully!")

        except ImportError:
            print(
                "ERROR: llama-cpp-python not installed. Run: pip install llama-cpp-python"
            )
            llm = "FAILED"
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            llm = "FAILED"


# ===============================
# System Prompt with Escalation Logic
# ===============================
SYSTEM_PROMPT = """Bạn là trợ lý du lịch TripConnect. Hãy trả lời thân thiện và hữu ích bằng tiếng Việt.

QUAN TRỌNG - Quy tắc bắt buộc:
Nếu khách hàng có ý định muốn gặp người thật, nhân viên hỗ trợ, hoặc có vấn đề cần giải quyết mà bạn không thể tự xử lý, bạn PHẢI thêm thẻ [ESCALATE] vào đầu câu trả lời của mình.

Các trường hợp PHẢI escalate (thêm [ESCALATE] phía trước):
1. Khách hàng yêu cầu: "gặp người thật", "gặp admin", "gặp nhân viên", "nói chuyện với người", "gặp nhà cung cấp", "gọi điện".
2. Khách hàng phàn nàn: "AI không hiểu", "trả lời sai rồi", "tôi muốn khiếu nại", "tại sao lâu thế".
3. Vấn đề phức tạp: Hoàn tiền, hủy đơn hàng, lỗi thanh toán, sự cố phát sinh khi đi tour.
4. Khi bạn không chắc chắn về thông tin hoặc khách hàng hỏi lặp đi lặp lại một vấn đề mà bạn chưa giải quyết được.

Khi bạn quyết định escalate:
- BẮT ĐẦU bằng thẻ: [ESCALATE]
- Sau đó nói: "Tôi đã hiểu yêu cầu của bạn. Tôi sẽ kết nối bạn với nhân viên hỗ trợ (Admin) ngay bây giờ. Vui lòng đợi trong giây lát, họ sẽ phản hồi bạn sớm nhất có thể ạ."

Ví dụ:
User: "cho tui nói chuyện với admin đi"
Assistant: [ESCALATE] Tôi đã hiểu yêu cầu của bạn. Tôi sẽ kết nối bạn với nhân viên hỗ trợ (Admin) ngay bây giờ. Vui lòng đợi trong giây lát, họ sẽ phản hồi bạn sớm nhất có thể ạ.
"""


@app.post("/v1/chat/completions")
async def chat_completions(data: ChatRequest):
    """OpenAI-compatible chat completions endpoint with escalation detection"""
    try:
        load_llm()

        if llm == "FAILED":
            raise Exception("LLM initialization failed - check logs for details")
        if llm == "NOT_FOUND":
            raise Exception(f"Model file not found at {GGUF_MODEL_PATH}")

        # Build prompt with system message
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        for msg in data.messages:
            prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        # Generate response
        output = llm(
            prompt,
            max_tokens=data.max_tokens,
            temperature=data.temperature,
            stop=["<|im_end|>", "<|im_start|>"],
            echo=False,
        )

        response_text = output["choices"][0]["text"].strip()

        # Check for escalation tag
        should_escalate = "[ESCALATE]" in response_text

        # Remove the tag from displayed response
        clean_response = response_text.replace("[ESCALATE]", "").strip()

        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": clean_response},
                    "finish_reason": output["choices"][0].get("finish_reason", "stop"),
                }
            ],
            "usage": output.get("usage", {}),
            "should_escalate": should_escalate,
        }

    except Exception as e:
        print(f"Chat Error: {e}")
        # Fallback response
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Tôi là trợ lý AI chuyên về du lịch. Hiện tại hệ thống đang bận, bạn vui lòng đợi nhân viên hỗ trợ hoặc thử lại sau nhé!",
                    }
                }
            ],
            "should_escalate": True,  # Escalate on error
        }


@app.post("/v1/embeddings/fuse")
async def generate_fused_vector(data: FusedVectorRequest):
    """Generate fused embedding vector for tour recommendation"""
    try:
        # 1. Text Embedding
        text_content = f"{data.title} {data.summary} {data.description} {data.address}"
        text_vector = text_model.encode(text_content)

        # 2. Image Embedding
        image_vectors = []
        for url in data.image_urls[:3]:  # Top 3 images
            try:
                response = requests.get(url, timeout=5)
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img_vector = image_model.encode(img)
                image_vectors.append(img_vector)
            except Exception as e:
                print(f"Error processing image {url}: {e}")

        if image_vectors:
            image_vector = np.mean(image_vectors, axis=0)
        else:
            image_vector = np.zeros(512)  # CLIP B-32 dimension

        # 3. Numeric Normalization (Min-Max scaling)
        price_norm = (data.price - data.min_price_global) / (
            data.max_price_global - data.min_price_global + 1e-6
        )
        duration_norm = data.duration_days / (data.max_duration_global + 1e-6)

        # 4. Concatenation: text (384) + image (512) + numeric (2) = 898 dims
        fused_vector = np.concatenate(
            [text_vector, image_vector, [float(price_norm), float(duration_norm)]]
        )

        return {
            "vector": fused_vector.tolist(),
            "dimensions": len(fused_vector),
            "insights": {
                "text_summary": data.summary[:100],
                "processed_images": len(image_vectors),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
