"""Integration tests for Research Vault.

These tests verify the full pipeline: upload -> extract -> review -> query.
All external services (LLM, embeddings, Qdrant) are mocked for deterministic results.
"""
