"""Dense text embedding module using local Sentence Transformers (Stage 5).

Provides single and batch embedding generation with L2 vector normalization
for semantic retrieval and vector store indexing.
"""

from abc import ABC, abstractmethod
import math
from typing import Optional, Sequence, Union
import numpy as np

from config.settings import get_settings
from core.logger import get_logger

logger = get_logger(__name__)


def l2_normalize(vector: Union[list[float], np.ndarray]) -> list[float]:
    """Normalize a vector to unit length (L2 norm = 1.0)."""
    arr = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0.0 or math.isnan(norm):
        return arr.tolist()
    return (arr / norm).tolist()


class BaseEmbedder(ABC):
    """Abstract interface for dense embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the produced dense embedding vectors."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generate a normalized dense embedding vector for a single text string."""
        pass

    @abstractmethod
    def embed_batch(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Generate normalized dense embedding vectors for a sequence of texts."""
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local Sentence Transformers embedding provider with automatic L2 normalization."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ) -> None:
        """Initialize the sentence transformer embedder.

        Args:
            model_name: HuggingFace or local model name/path.
                Defaults to setting embedding_model_name ("all-MiniLM-L6-v2").
            device: Compute device ('cpu', 'cuda', etc.). Defaults to settings.
            normalize_embeddings: Whether to return L2-normalized unit vectors.
        """
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model_name
        self.device = device or settings.embedding_device
        self.normalize_embeddings = normalize_embeddings

        self._model = None
        self._dimension: Optional[int] = None

    @property
    def model(self):
        """Lazily load and return the SentenceTransformer model instance."""
        if self._model is None:
            logger.info(
                "Loading SentenceTransformer model '%s' on device '%s'...",
                self.model_name,
                self.device,
            )
            from sentence_transformers import SentenceTransformer

            try:
                self._model = SentenceTransformer(
                    model_name_or_path=self.model_name,
                    device=self.device,
                    local_files_only=True,
                )
            except Exception:
                self._model = SentenceTransformer(
                    model_name_or_path=self.model_name,
                    device=self.device,
                )
            if hasattr(self._model, "get_embedding_dimension"):
                self._dimension = self._model.get_embedding_dimension()
            else:
                self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(
                "Loaded '%s' successfully. Embedding dimension: %d",
                self.model_name,
                self._dimension,
            )
        return self._model

    @property
    def dimension(self) -> int:
        """Vector dimensionality (e.g., 384 for all-MiniLM-L6-v2)."""
        if self._dimension is None:
            _ = self.model  # Triggers lazy load
        return self._dimension or 384

    def embed_text(self, text: str) -> list[float]:
        """Generate a dense embedding vector for a single text string.

        Args:
            text: Input string to embed.

        Returns:
            Normalized list of floats representing the dense vector.
        """
        # Encode single text
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )

        vector = embedding.tolist()
        if self.normalize_embeddings:
            return l2_normalize(vector)
        return vector

    def embed_batch(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Generate dense embedding vectors for a batch of text strings.

        Args:
            texts: Sequence of input strings to embed.
            batch_size: Mini-batch size for model inference.

        Returns:
            List of normalized dense vectors (list of float lists).
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )

        vectors = [vec.tolist() for vec in embeddings]
        if self.normalize_embeddings:
            return [l2_normalize(v) for v in vectors]
        return vectors
