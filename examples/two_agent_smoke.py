"""Connect two BBS agents and capture their initial observations."""

from bbs_gym.env import BbsGym


def main() -> None:
    with BbsGym() as gym:
        for agent_id in ("agent-001", "agent-002"):
            agent = gym.connect(agent_id)
            print(f"--- {agent_id} ---")
            print(agent.observe(seconds=3.0)[-1200:])


if __name__ == "__main__":
    main()

