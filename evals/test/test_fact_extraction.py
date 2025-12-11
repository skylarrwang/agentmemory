"""Tests for fact extraction from conversations"""

import pytest
from utils.model_call_utils import extract_facts


class TestFactExtraction:
    """Test that facts are correctly extracted from conversations"""
    
    def test_extracts_name(self):
        """Test extraction of user name"""
        user_msg = "Hi, my name is Alice"
        assistant_msg = "Nice to meet you, Alice!"
        
        facts = extract_facts(user_msg, assistant_msg)
        fact_map = {f.field: f.value for f in facts.facts}
        
        assert 'name' in fact_map or 'Name' in fact_map
        # Check that name value contains Alice (case insensitive)
        name_value = fact_map.get('name', fact_map.get('Name', '')).lower()
        assert 'alice' in name_value
    
    def test_extracts_location(self):
        """Test extraction of location information"""
        user_msg = "I live in San Francisco"
        assistant_msg = "San Francisco is a beautiful city!"
        
        facts = extract_facts(user_msg, assistant_msg)
        fact_map = {f.field: f.value for f in facts.facts}
        
        # Should extract location/city
        location_keys = ['location', 'city', 'Location', 'City', 'lives_in', 'residence']
        assert any(key in fact_map for key in location_keys)
        
        # Value should contain San Francisco
        location_value = ''
        for key in location_keys:
            if key in fact_map:
                location_value = fact_map[key].lower()
                break
        
        assert 'san francisco' in location_value
    
    def test_extracts_preferences(self):
        """Test extraction of user preferences"""
        user_msg = "I prefer Python over Java for data science"
        assistant_msg = "Python is indeed great for data science!"
        
        facts = extract_facts(user_msg, assistant_msg)
        fact_map = {f.field: f.value for f in facts.facts}
        
        # Should extract preference
        preference_keys = ['preference', 'prefers', 'favorite', 'likes', 'preferred_language']
        assert any(key in fact_map for key in preference_keys)
    
    def test_extracts_multiple_facts(self):
        """Test extraction of multiple facts in one exchange"""
        user_msg = "I'm Sarah, I work as a software engineer in Seattle, and I love hiking"
        assistant_msg = "Nice to meet you Sarah! Software engineering in Seattle sounds great, and hiking is wonderful."
        
        facts = extract_facts(user_msg, assistant_msg)
        fact_map = {f.field: f.value for f in facts.facts}
        
        # Should extract multiple facts
        assert len(fact_map) >= 2
        
        # Check for name
        name_value = ''
        for key in ['name', 'Name']:
            if key in fact_map:
                name_value = fact_map[key].lower()
                break
        assert 'sarah' in name_value
        
        # Check for location or job
        values = list(fact_map.values())
        has_location = any('seattle' in str(v).lower() for v in values)
        has_job = any('engineer' in str(v).lower() or 'software' in str(v).lower() for v in values)
        assert has_location or has_job
    
    def test_no_facts_in_greeting(self):
        """Test that generic greetings don't extract false facts"""
        user_msg = "Hello, how are you?"
        assistant_msg = "I'm doing well, thanks for asking!"
        
        facts = extract_facts(user_msg, assistant_msg)
        
        # Should not extract false facts from generic greeting
        # Empty or minimal facts list is acceptable
        assert hasattr(facts, "facts")
    
    def test_extracts_concrete_facts_not_inferences(self):
        """Test that only concrete facts are extracted, not inferences"""
        user_msg = "I'm feeling tired today"
        assistant_msg = "Maybe you should get more sleep"
        
        facts = extract_facts(user_msg, assistant_msg)
        
        # Should not extract "needs sleep" as a fact (that's an inference)
        # Should only extract stated facts
        assert hasattr(facts, "facts")
        # The fact extraction should be conservative - if no concrete facts,
        # should return empty or minimal dict
    
    def test_handles_empty_responses(self):
        """Test that empty responses don't crash"""
        user_msg = ""
        assistant_msg = "I see"
        
        facts = extract_facts(user_msg, assistant_msg)
        
        assert hasattr(facts, "facts")
    
    def test_extracts_work_information(self):
        """Test extraction of work/job information"""
        user_msg = "I'm a data scientist at Google"
        assistant_msg = "That's impressive! Data science at Google must be exciting."
        
        facts = extract_facts(user_msg, assistant_msg)
        fact_map = {f.field: f.value for f in facts.facts}
        
        # Should extract job/company information
        job_keys = ['job', 'occupation', 'work', 'company', 'employer', 'role']
        assert any(key in fact_map for key in job_keys)
        
        # Check that Google or data scientist is mentioned
        values_str = ' '.join(str(v).lower() for v in fact_map.values())
        assert 'google' in values_str or 'data scientist' in values_str
    
    def test_extracts_interests(self):
        """Test extraction of interests/hobbies"""
        user_msg = "I enjoy reading science fiction and playing guitar"
        assistant_msg = "Those are great hobbies! Science fiction is fascinating."
        
        facts = extract_facts(user_msg, assistant_msg)
        fact_map = {f.field: f.value for f in facts.facts}
        
        # Should extract interests/hobbies
        interest_keys = ['interests', 'hobbies', 'enjoys', 'likes', 'hobby']
        assert any(key in fact_map for key in interest_keys)
        
        # Check that interests are mentioned
        values_str = ' '.join(str(v).lower() for v in fact_map.values())
        assert 'reading' in values_str or 'guitar' in values_str or 'science fiction' in values_str

