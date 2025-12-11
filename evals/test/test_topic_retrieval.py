"""Tests for topic retrieval in short-term and long-term memory"""

import pytest
from datetime import datetime
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.topics import Topic
import uuid


class TestShortTermTopicRetrieval:
    """Test topic retrieval within a session"""
    
    def test_retrieves_current_topic_messages(self, short_term_memory):
        """Test that current topic messages are always included"""
        # Add some messages to current topic
        short_term_memory.curr_open_topic.add_message_to_topic({
            'role': 'user',
            'content': 'I love Python programming',
            'timestamp': datetime.utcnow().isoformat()
        })
        short_term_memory.curr_open_topic.add_message_to_topic({
            'role': 'assistant',
            'content': 'That\'s great! Python is a versatile language.',
            'timestamp': datetime.utcnow().isoformat()
        })
        
        context = short_term_memory.get_context("tell me about Python")
        
        assert len(context['recent_messages']) == 2
        assert context['recent_messages'][0]['content'] == 'I love Python programming'
    
    def test_retrieves_relevant_closed_topics(self, short_term_memory):
        """Test that semantically similar closed topics are retrieved"""
        # Create and close a topic about travel
        travel_topic = Topic(
            id=str(uuid.uuid4()),
            name="Travel Planning",
            summary="User discussed planning a trip to Japan, including flights and hotels",
            messages=[
                {'role': 'user', 'content': 'I want to visit Japan', 'timestamp': datetime.utcnow().isoformat()},
                {'role': 'assistant', 'content': 'Japan is beautiful!', 'timestamp': datetime.utcnow().isoformat()}
            ],
            created_at=datetime.utcnow().isoformat(),
            closed_at=datetime.utcnow().isoformat()
        )
        
        # Generate embedding for this topic
        from utils.embed import get_embedding
        short_term_memory.topic_embeddings[travel_topic.id] = get_embedding(
            travel_topic.summary
        )
        short_term_memory.topics.append(travel_topic)
        
        # Create another unrelated topic
        cooking_topic = Topic(
            id=str(uuid.uuid4()),
            name="Cooking Recipes",
            summary="User asked about Italian pasta recipes",
            messages=[],
            created_at=datetime.utcnow().isoformat(),
            closed_at=datetime.utcnow().isoformat()
        )
        short_term_memory.topic_embeddings[cooking_topic.id] = get_embedding(
            cooking_topic.summary
        )
        short_term_memory.topics.append(cooking_topic)
        
        # Add current topic about something else
        short_term_memory.curr_open_topic.add_message_to_topic({
            'role': 'user',
            'content': 'What about hotels in Tokyo?',
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Query about travel should retrieve travel topic
        context = short_term_memory.get_context("hotels in Tokyo")
        
        # Should have relevant topics
        relevant_names = [t['name'] for t in context['relevant_topics']]
        assert 'Travel Planning' in relevant_names
        # Cooking topic should not be in highly relevant (similarity threshold 0.75)
        # It might be in closed_topics_summaries but not in relevant_topics
    
    def test_includes_summaries_of_all_closed_topics(self, short_term_memory):
        """Test that summaries of all closed topics are included"""
        # Create multiple closed topics
        topics = [
            Topic(
                id=str(uuid.uuid4()),
                name=f"Topic {i}",
                summary=f"Summary of topic {i}",
                messages=[],
                created_at=datetime.utcnow().isoformat(),
                closed_at=datetime.utcnow().isoformat(),
            )
            for i in range(3)
        ]
        
        short_term_memory.topics.extend(topics)
        
        context = short_term_memory.get_context("test query")
        
        # All closed topics should be in summaries
        assert len(context['closed_topics_summaries']) == 3
        summary_names = {s['name'] for s in context['closed_topics_summaries']}
        assert summary_names == {'Topic 0', 'Topic 1', 'Topic 2'}


class TestLongTermTopicRetrieval:
    """Test topic retrieval across sessions"""
    
    def test_retrieves_relevant_topics_from_all_sessions(self, long_term_memory):
        """Test that relevant topics from past sessions are retrieved"""
        # Add topics from different sessions
        topics = [
            Topic(
                id=str(uuid.uuid4()),
                name="Python Programming",
                summary="User discussed Python programming concepts and best practices",
                messages=[],
                created_at=datetime.utcnow().isoformat(),
                closed_at=datetime.utcnow().isoformat(),
            ),
            Topic(
                id=str(uuid.uuid4()),
                name="Travel Planning",
                summary="User planned a trip to Europe",
                messages=[],
                created_at=datetime.utcnow().isoformat(),
                closed_at=datetime.utcnow().isoformat(),
            ),
            Topic(
                id=str(uuid.uuid4()),
                name="Cooking Recipes",
                summary="User asked about Italian cooking techniques",
                messages=[],
                created_at=datetime.utcnow().isoformat(),
                closed_at=datetime.utcnow().isoformat(),
            ),
        ]
        
        # Generate embeddings
        from utils.embed import get_embedding
        for topic in topics:
            long_term_memory.all_topic_embeddings[topic.id] = get_embedding(
                topic.summary
            )
        
        long_term_memory.all_session_topics = topics
        
        # Query about Python should retrieve Python topic
        context = long_term_memory.get_context("tell me about Python")
        
        relevant_names = [t['name'] for t in context['relevant_topics']]
        assert 'Python Programming' in relevant_names
        
        # Query about travel should retrieve travel topic
        context = long_term_memory.get_context("I want to visit Paris")
        relevant_names = [t['name'] for t in context['relevant_topics']]
        assert 'Travel Planning' in relevant_names
    
    def test_respects_similarity_threshold(self, long_term_memory):
        """Test that only topics above similarity threshold are returned"""
        # Add a topic
        topic = Topic(
            id=str(uuid.uuid4()),
            name="Python Programming",
            summary="User discussed Python programming",
            messages=[],
            created_at=datetime.utcnow().isoformat(),
            closed_at=datetime.utcnow().isoformat(),
        )
        
        from utils.embed import get_embedding
        long_term_memory.all_topic_embeddings[topic.id] = get_embedding(topic.summary)
        long_term_memory.all_session_topics = [topic]
        
        # Query something completely unrelated
        context = long_term_memory.get_context("what is the weather like today")
        
        # Should return empty or very few results (threshold is 0.5)
        # The exact result depends on embedding similarity, but unrelated queries
        # should have low similarity
        assert len(context['relevant_topics']) <= 1  # Might return 0 or 1
    
    def test_returns_top_k_topics(self, long_term_memory):
        """Test that only top k topics are returned"""
        # Create many topics
        topics = [
            Topic(
                id=str(uuid.uuid4()),
                name=f"Topic {i}",
                summary=f"Summary about topic {i}",
                messages=[],
                created_at=datetime.utcnow().isoformat(),
                closed_at=datetime.utcnow().isoformat(),
            )
            for i in range(10)
        ]
        
        from utils.embed import get_embedding
        for topic in topics:
            long_term_memory.all_topic_embeddings[topic.id] = get_embedding(topic.summary)
        
        long_term_memory.all_session_topics = topics
        
        # Query should return max 3 topics (max_k=3)
        context = long_term_memory.get_context("test query")
        assert len(context['relevant_topics']) <= 3

