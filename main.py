#!/usr/bin/env python3
"""CLI entry point for the conversational memory agent"""

import sys
from agent import Agent

def main():
    """Main CLI entry point"""
    # Get username as first argument (required)
    if len(sys.argv) < 2:
        print("Usage: python main.py <username>")
        print("\nExample: python main.py alice")
        sys.exit(1)
    
    username = sys.argv[1].strip()
    
    if not username:
        print("Error: Username cannot be empty")
        sys.exit(1)
    
    print(f"Logged in as: {username}")
    print(f"Session storage: sessions/{username}.*/")
    print(f"Long-term memory: longterm_memory/{username}/")
    print("\n" + "="*50)
    print("Conversational Memory Agent")
    print("Type 'exit' or 'quit' to end the session")
    print("="*50 + "\n")
    
    # Initialize agent with username
    agent = Agent(username)
    
    # Interactive chat loop
    try:
        while True:
            user_input = input(f"{username}> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nEnding session...")
                agent.end_session()
                print("Session saved. Goodbye!")
                break
            
            # Get response from agent
            response = agent.single_turn_chat(user_input)
            print(f"\nAssistant: {response}\n")
    
    except KeyboardInterrupt:
        print("\n\nSession interrupted. Saving...")
        agent.end_session()
        print("Session saved. Goodbye!")
    except Exception as e:
        print(f"\nError: {e}")
        agent.end_session()
        sys.exit(1)

if __name__ == "__main__":
    main()

