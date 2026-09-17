# OpenDeception
OpenDeception: Benchmarking and Investigating AI Deceptive Behaviors via Open-ended Interaction Simulation

# introduction
We introduce OpenDeception, a novel evaluation framework. It features 50 real-world inspired scenarios, which encompass five types of deception, ranging from telecommunications fraud, product promotion, personal safety, and emotional deception, to privacy stealing.

# code
We construct 50 different deception scenarios in `example.py` and then use `simulate.py` to generate dialogue data. In `simulate.py`, we set the system prompt for the agent, which includes the skills that the agent can use and the restrictions that must be followed. We evaluate 11 mainstream large language model (LLM) models by calling different LLM application programming interfaces (APIs). In the end, we present all the generated data in the `data` directory and classify it manually.

## Brief Introduction to the Functions and Roles of Each File:

### examples.py:

Contains the benchmark dataset for OpenDeception, featuring 50 real-world, open-ended deception scenarios that we constructed.

### simulate.py: 

The core file responses for implementing dialogue simulation. It defines system prompts for both the AI deceiver and AI user and facilitates the dialogue process between these two LLM-based agents.

### agent.py: 

Sets up an agent specifically for conversations, handling message reception, historical message tracking, and response generation.

### utils.py: 

Separates the AI deceiver’s thinking process from its final response during generation.

### llm.py: 

Specifies the LLM class used in simulated conversations.

### com_generate.py: 

Manages error handling for generated messages.

### llama.py, qwen.py, and gpt.py:

Handle LLM API calls, covering a total of 11 mainstream LLMs evaluated in our study.

# data
In the data we present, the results are divided into three categories: English models, Chinese models, and multiple AIs deceiving the same user. The results of each type of data are also subdivided. "Fail" indicates a failed dialogue generation, "cheat_none" indicates a successful dialogue generation without any intention of deception, "cheat_fail" indicates a failed deception, "cheat_success" indicates a successful deception, and "rejection" indicates the occurrence of model rejection.
