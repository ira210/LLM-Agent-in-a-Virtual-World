ntern Challenge: LLM Agent in a Virtual World

Build a system that places an LLM agent into a virtual world where it can perceive its environment, take actions, and accomplish goals. The core challenge isn't the world itself — it's the harness: the interface between an intelligent agent and an environment it can act in.

At a minimum, your system should:
• Create a virtual environment the agent can exist in (2D grid, 3D scene, text-based world — your choice)
• Define an observation format that represents the agent's current state and surroundings
• Define an action space the agent can use to interact with the world (e.g. move, turn, look, pick up)
• Wire up an LLM (e.g. Claude, GPT, or any model with an API) to observe state, reason, and choose actions in a loop
• Demonstrate the agent completing at least one goal-directed task (e.g. "go to the red cube", "find the key and open the door", "explore and describe the room")

We're intentionally leaving the approach open — use any tools, frameworks, or libraries you find effective.

What to submit:
• A working codebase as a public GitHub repository
• A README with instructions to run your system
• Example input(s) and output(s) — ideally a recording or log of the agent acting in the world
• (Optional) A short note on your design choices: how you represent observations, why you chose your action space, what worked and what didn't

What we care about:
• Quality of the agent harness — how well you've designed the interface between the LLM and the environment
• Whether the agent can actually accomplish tasks, not just generate plausible text
• Thoughtfulness about observation representation — what does the agent need to know, and how do you tell it?
• Creativity in the world, the tasks, or the agent's capabilities
• Simplicity and usability of your solution
