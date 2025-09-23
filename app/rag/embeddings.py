from abc import ABC, abstractmethod
from typing import List


try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass




class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "all-mpnet-base-v2"):  # High-quality semantic understanding model
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not available.")

        import torch
        # Determine best device
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'  # Apple Silicon acceleration
        else:
            device = 'cpu'

        self.model = SentenceTransformer(model_name, device=device)

        # Set to model's actual maximum (all-mpnet-base-v2 supports up to 514 tokens)
        self.model.max_seq_length = 512  # Safe limit to avoid position embedding errors

        print(f"Initialized LocalEmbeddingProvider with {model_name} on {device}")

    def embed_text(self, text: str) -> List[float]:
        embedding = self.model.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        return embedding.cpu().numpy().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Batch processing with optimizations
        embeddings = self.model.encode(
            texts,
            batch_size=32,  # Process in batches for speed
            show_progress_bar=len(texts) > 10,  # Show progress for large batches
            convert_to_tensor=True,  # Keep as tensors during processing
            normalize_embeddings=True  # Normalize for better similarity search
        )
        return embeddings.cpu().numpy().tolist()  # Convert to CPU then to list




def get_embedding_provider() -> EmbeddingProvider:
    # Use local embeddings only
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            return LocalEmbeddingProvider()
        except Exception as e:
            print(f"Local embeddings failed: {e}.")
            raise RuntimeError("Local embeddings failed. Install sentence-transformers.")

    # No fallback - require sentence transformers
    raise RuntimeError("sentence-transformers not available. Install with: pip install sentence-transformers")
