"""
Unit tests for SemanticChunker.
"""

from app.services.semantic_chunker import SemanticChunker


def test_chunker_short_text_no_split():
    chunker = SemanticChunker(max_words=18, soft_limit_words=12, min_prefix_words=6)
    prefix, remaining = chunker.process_interim("hola mundo esto es una prueba corta", lang="es")
    assert prefix is None
    assert remaining == "hola mundo esto es una prueba corta"


def test_chunker_splits_on_connector_in_spanish():
    chunker = SemanticChunker(max_words=18, soft_limit_words=12, min_prefix_words=6)
    
    # Text with 16 words and a connector " pero "
    full_text = "en el día de hoy vamos a probar esta aplicación pero la latencia sigue siendo muy baja"
    prefix, remaining = chunker.process_interim(full_text, lang="es")
    
    assert prefix is not None
    assert "pero" in prefix
    assert prefix == "en el día de hoy vamos a probar esta aplicación pero"
    assert remaining == "la latencia sigue siendo muy baja"


def test_chunker_splits_on_punctuation_in_english():
    chunker = SemanticChunker(max_words=18, soft_limit_words=12, min_prefix_words=6)
    
    full_text = "in this video we are building real time transcription. it works amazingly well with low latency"
    prefix, remaining = chunker.process_interim(full_text, lang="en")
    
    assert prefix is not None
    assert prefix == "in this video we are building real time transcription."
    assert remaining == "it works amazingly well with low latency"


def test_chunker_hard_limit_word_boundary():
    chunker = SemanticChunker(max_words=18, soft_limit_words=12, min_prefix_words=6)
    
    # 20 words without any connectors
    words = [f"word{i}" for i in range(20)]
    full_text = " ".join(words)
    
    prefix, remaining = chunker.process_interim(full_text, lang="en")
    assert prefix is not None
    assert len(prefix.split()) == 17
    assert len(remaining.split()) == 3


def test_chunker_process_final():
    chunker = SemanticChunker(max_words=18, soft_limit_words=12, min_prefix_words=6)
    
    # Simulate first split
    text1 = "en el día de hoy vamos a probar esta aplicación pero la latencia sigue siendo baja"
    prefix, remaining = chunker.process_interim(text1, lang="es")
    assert prefix is not None
    
    # Now speech finishes with final
    final_full = "en el día de hoy vamos a probar esta aplicación pero la latencia sigue siendo baja finalizada"
    remainder = chunker.process_final(final_full)
    
    assert remainder == "la latencia sigue siendo baja finalizada"
    assert chunker.committed_text == ""
