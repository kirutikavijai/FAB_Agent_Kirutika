#!/usr/bin/env python3
import sys
from core.data_loader import load_all_data
from core.agent import run_agent, run_evaluation

def main():
    data, all_years = load_all_data()

    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            '  python main.py "your question"\n'
            "Or to run built-in test suite:\n"
            "  python main.py --eval\n"
        )
        sys.exit(0)

    first_arg = sys.argv[1]

    if first_arg in ("--eval", "--demo"):
        # evaluation: no plots
        run_evaluation(data, all_years, show_plots=False)
        sys.exit(0)

    # Normal single-question mode
    question = " ".join(sys.argv[1:])
    run_agent(question, data, all_years, show_plots=True)

if __name__ == "__main__":
    main()
