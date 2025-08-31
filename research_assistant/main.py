#!/usr/bin/env python
from research_assistant.crew import ResearchAssistantCrew
from futureagi_integration import evaluate_text

def run():
    # Inputs passed to CrewAI task
    inputs = {
        'topic': 'Quantum Computing in Finance'
    }
    
    # Run CrewAI
    crew = ResearchAssistantCrew().crew()
    result = crew.kickoff(inputs=inputs)

    print("CrewAI Result:\n", result)

    # Evaluate with Future AGI
    evaluation = evaluate_text(result)
    print("\nFuture AGI Evaluation:\n", evaluation)

if __name__ == "__main__":
    run()
