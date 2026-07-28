# tty-agent

Reusable terminal-agent runtime for LLM-driven interactive TTY sessions.

This package contains the generic core from the Spree workspace: terminal
actions, model adapters, memory, prompt modules, pyte-backed observations,
activity runners, evaluator-owned metric collection, and telnet/rlogin/PTY
transports. Evaluation profiles can passively extract metrics from normal
observations and perform a separate final probe without consuming an agent
decision tick or adding the probe response to model context.
