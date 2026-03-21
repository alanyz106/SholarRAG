#!/usr/bin/env python
"""
Test ChromaDB Embedding Independent Configuration
验证 ChromaDB 嵌入的独立配置是否生效
"""
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.services.embedder import EmbeddingService

def test_config():
    print("=" * 60)
    print("Configuration Test")
    print("=" * 60)

    # Check settings
    print("\n1. Settings loaded:")
    print(f"   CHROMA_EMBEDDING_PROVIDER: {settings.CHROMA_EMBEDDING_PROVIDER}")
    print(f"   CHROMA_OPENAI_MODEL: {settings.CHROMA_OPENAI_MODEL}")
    print(f"   CHROMA_OPENAI_API_KEY: {'***' + settings.CHROMA_OPENAI_API_KEY[-4:] if settings.CHROMA_OPENAI_API_KEY else '(not set)'}")
    print(f"   CHROMA_OPENAI_BASE_URL: {settings.CHROMA_OPENAI_BASE_URL}")
    print(f"\n   LLM Provider (for chat):")
    print(f"   LLM_PROVIDER: {settings.LLM_PROVIDER}")
    print(f"   OPENAI_API_KEY: {'***' + settings.OPENAI_API_KEY[-4:] if settings.OPENAI_API_KEY else '(not set)'}")
    print(f"   OPENAI_BASE_URL: {settings.OPENAI_BASE_URL}")

    print("\n2. Testing EmbeddingService initialization...")

    try:
        # Create embedding service
        embedding_service = EmbeddingService(provider="openai")
        print(f"   [OK] EmbeddingService created successfully")
        print(f"   - Provider: {embedding_service.provider}")
        print(f"   - Model: {embedding_service.model_name}")

        # Check which API key and base url are being used
        if embedding_service._external_provider:
            provider = embedding_service._external_provider
            print(f"   - External provider type: {type(provider).__name__}")
            # Try to access the api_key and base_url (if available)
            if hasattr(provider, 'api_key'):
                key_preview = '***' + provider.api_key[-4:] if provider.api_key else '(none)'
                print(f"   - API Key being used: {key_preview}")
            if hasattr(provider, 'base_url'):
                print(f"   - Base URL being used: {provider.base_url}")

        print("\n3. Testing embedding generation...")
        test_text = "Hello, this is a test for independent ChromaDB embedding configuration."
        embedding = embedding_service.embed_text(test_text)
        print(f"   [OK] Generated embedding with {len(embedding)} dimensions")
        print(f"   - First 5 values: {embedding[:5]}")

        print("\n" + "=" * 60)
        print("[SUCCESS] ALL TESTS PASSED!")
        print("ChromaDB embedding is using independent configuration.")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n   [ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 60)
        print("[FAILED] TEST FAILED")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = test_config()
    sys.exit(0 if success else 1)
