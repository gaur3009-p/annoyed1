from llm_client import generate_text
from agents.strategy_agent import build_strategy_prompt
from agents.copy_agent import build_copy_prompt


def run_agents(campaign):
    strategy_raw = generate_text(build_strategy_prompt(campaign))
    strategy = strategy_raw.split("### STRATEGY OUTPUT")[-1].strip()

    copy_raw = generate_text(build_copy_prompt(strategy, campaign))
    copy = copy_raw.split("### COPY OUTPUT")[-1].strip()

    return strategy, copy
