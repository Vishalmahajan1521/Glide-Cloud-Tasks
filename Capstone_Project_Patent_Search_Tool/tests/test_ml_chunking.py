"""
Tests for ML chunking functions
"""
import pytest
from app.ml.chunking import (
    split_into_sections,
    create_chunks,
    sliding_window_chunk,
    section_weight
)


class TestSplitIntoSections:
    """Tests for split_into_sections function"""
    
    def test_split_with_all_sections(self):
        """Test splitting text with all sections present"""
        text = """
        Abstract
        This is the abstract text.
        
        Description
        This is the description text.
        
        Claims
        This is the claims text.
        """
        
        sections = split_into_sections(text)
        
        assert "abstract" in sections
        assert "description" in sections
        assert "claim" in sections
        assert "abstract" in sections["abstract"].lower()
        assert "description" in sections["description"].lower()
    
    def test_split_with_partial_sections(self):
        """Test splitting text with only some sections"""
        text = """
        Abstract
        This is the abstract text.
        
        Description
        This is the description text.
        """
        
        sections = split_into_sections(text)
        
        assert "abstract" in sections
        assert "description" in sections
        assert "claim" not in sections
    
    def test_split_no_sections(self):
        """Test splitting text with no section markers"""
        text = "This is just plain text with no sections."
        
        sections = split_into_sections(text)
        
        assert len(sections) == 0
    
    def test_split_case_insensitive(self):
        """Test that section detection is case insensitive"""
        text = """
        ABSTRACT
        This is the abstract.
        
        Detailed Description
        This is the description.
        """
        
        sections = split_into_sections(text)
        
        assert "abstract" in sections
        assert "description" in sections


class TestSlidingWindowChunk:
    """Tests for sliding_window_chunk function"""
    
    def test_chunk_basic(self):
        """Test basic chunking"""
        text = " ".join(["word"] * 100)  # 100 words
        chunks = sliding_window_chunk(text, chunk_size=20, overlap=5)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_chunk_empty_text(self):
        """Test chunking empty text"""
        chunks = sliding_window_chunk("")
        assert chunks == []
    
    def test_chunk_whitespace_only(self):
        """Test chunking whitespace-only text"""
        chunks = sliding_window_chunk("   \n\t  ")
        assert chunks == []
    
    def test_chunk_small_text(self):
        """Test chunking text smaller than chunk_size"""
        text = "This is a short text."
        # Use overlap < chunk_size to avoid infinite loop
        chunks = sliding_window_chunk(text, chunk_size=100, overlap=20)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_overlap(self):
        """Test that chunks have proper overlap"""
        text = " ".join([f"word{i}" for i in range(50)])
        chunks = sliding_window_chunk(text, chunk_size=10, overlap=3)
        
        # Should have multiple chunks due to overlap
        assert len(chunks) > 1
        
        # First few words of second chunk should overlap with end of first chunk
        first_chunk_words = chunks[0].split()
        second_chunk_words = chunks[1].split()
        
        # Check overlap (last 3 words of first should be in first 3 of second)
        assert first_chunk_words[-3:] == second_chunk_words[:3]


class TestCreateChunks:
    """Tests for create_chunks function"""
    
    def test_create_chunks_with_sections(self):
        """Test creating chunks from sections"""
        sections = {
            "abstract": "This is the abstract text. " * 10,
            "description": "This is the description text. " * 20,
            "claim": "Claim 1: A system. " * 5
        }
        
        chunks = create_chunks(sections)
        
        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
        assert all("chunk_type" in chunk for chunk in chunks)
        assert all("section_priority" in chunk for chunk in chunks)
        assert all("chunk_index" in chunk for chunk in chunks)
    
    def test_create_chunks_fallback(self):
        """Test fallback when no sections detected"""
        sections = {}
        full_text = "This is some text without section markers. " * 10
        
        chunks = create_chunks(sections, full_text=full_text)
        
        assert len(chunks) > 0
        # All chunks should be type "description" in fallback
        assert all(chunk["chunk_type"] == "description" for chunk in chunks)
    
    def test_create_chunks_empty(self):
        """Test creating chunks from empty sections and text"""
        chunks = create_chunks({}, full_text="")
        
        # Should return empty list or handle gracefully
        assert isinstance(chunks, list)
    
    def test_create_chunks_chunk_types(self):
        """Test that chunks have correct chunk_type"""
        sections = {
            "abstract": "Abstract text " * 10,
            "claim": "Claim text " * 10
        }
        
        chunks = create_chunks(sections)
        
        chunk_types = [chunk["chunk_type"] for chunk in chunks]
        assert "abstract" in chunk_types
        assert "claim" in chunk_types


class TestSectionWeight:
    """Tests for section_weight function"""
    
    def test_section_weights(self):
        """Test section weight values"""
        assert section_weight("claim") == 1.0
        assert section_weight("abstract") == 0.7
        assert section_weight("description") == 0.4
    
    def test_section_weight_unknown(self):
        """Test section weight for unknown section"""
        assert section_weight("unknown") == 0.4  # default
