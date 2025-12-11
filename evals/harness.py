#!/usr/bin/env python3
"""simple harness for manual testing of memory agent functionality"""

import sys
from pathlib import Path

# add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import Agent

def test_short_term_memory():
    """test 1: short-term memory + fact storage"""
    print("\n" + "="*60)
    print("TEST 1: Short-Term Memory + Fact Storage")
    print("="*60)
    
    agent = Agent("test_user")
    
    queries = [
        "Hi, my name is Alice and I'm a data scientist",
        "I work with machine learning models at Google",
        "I'm currently working on a project using TensorFlow",
        "It's been challenging but really interesting",
        "Can you help me understand how to optimize it?",
    ]
    
    print("\nrunning queries...")
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}] user: {query}")
        response = agent.single_turn_chat(query)
        print(f"     assistant: {response[:300]}...")
    
    print("\nchecking facts...")
    facts = agent.memory.long_term.facts
    print(f"  stored facts: {len(facts)}")
    for field, data in facts.items():
        print(f"    {field}: {data['value']} (importance: {data['importance']})")
    
    print("\nending session...")
    agent.end_session()
    print("✓ test 1 complete")

def test_long_term_memory():
    """test 2: long-term memory retrieval"""
    print("\n" + "="*60)
    print("TEST 2: Long-Term Memory Retrieval")
    print("="*60)
    
    agent = Agent("test_user")
    
    queries = [
        "What's my name?",
        "Where do I work?",
        "Can you remind me about that TensorFlow project I was working on?",
    ]
    
    print("\nrunning queries...")
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}] user: {query}")
        response = agent.single_turn_chat(query)
        print(f"     assistant: {response[:300]}...")
    
    print("\nchecking stored data...")
    print(f"  facts: {len(agent.memory.long_term.facts)}")
    print(f"  topics: {len(agent.memory.long_term.all_session_topics)}")
    
    print("\nending session...")
    agent.end_session()
    print("✓ test 2 complete")

def test_topic_switching():
    """test 3: in-session topic switching"""
    print("\n" + "="*60)
    print("TEST 3: In-Session Topic Switching")
    print("="*60)
    
    agent = Agent("test_user")
    
    queries = [
        "I want to learn about neural networks and deep learning. Give me a brief overview of the topic.",
        "Can you explain backpropagation in one sentence?",
        "How does gradient descent work with backpropagation? Be brief.",
        "What are the differences between different activation functions? Give me 3 main ones.",
        "Actually I want to go to Waffle House right now",
        "What are the best items to order at Waffle House?",
        "How does their hash browns preparation work?",
        "Going back to neural networks, what were the questions I had before that I was confused on?",
    ]
    
    print("\nrunning queries...")
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}] user: {query}")
        response = agent.single_turn_chat(query)
        print(f"     assistant: {response[:300]}...")
        
        # check for topic switches
        if i > 1:
            closed_topics = len(agent.memory.short_term.topics)
            if closed_topics > 0:
                print(f"     [closed topics: {closed_topics}]")
    
    print("\nchecking topics...")
    print(f"  closed topics in session: {len(agent.memory.short_term.topics)}")
    for i, topic in enumerate(agent.memory.short_term.topics, 1):
        print(f"    topic {i}: {topic.name}")
        print(f"      summary: {topic.summary[:80]}...")
    
    print("\nending session...")
    agent.end_session()
    print("✓ test 3 complete")

def main():
    """run all tests"""
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()
        if test_name == "1" or test_name == "short":
            test_short_term_memory()
        elif test_name == "2" or test_name == "long":
            test_long_term_memory()
        elif test_name == "3" or test_name == "switch":
            test_topic_switching()
        else:
            print(f"unknown test: {test_name}")
            print("usage: python evals/harness.py [1|2|3|short|long|switch]")
            sys.exit(1)
    else:
        # run all tests
        test_short_term_memory()
        test_long_term_memory()
        test_topic_switching()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)
    print("\ncheck results in:")
    print("  - longterm_memory/test_user/facts.json")
    print("  - longterm_memory/test_user/all_session_topics.json")
    print("  - sessions/test_user.*/topics.json")

if __name__ == "__main__":
    main()

