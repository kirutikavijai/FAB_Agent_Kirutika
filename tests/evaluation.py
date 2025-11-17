from core.data_loader import load_all_data
from core.agent import run_evaluation

if __name__ == "__main__":
    data, all_years = load_all_data()
    run_evaluation(data, all_years, show_plots=False)
