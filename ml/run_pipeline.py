import run_clean
import run_explain
import run_train


def main() -> None:
    print("Step 1/3: cleaning data...")
    run_clean.main()
    print("Step 2/3: training and selecting best model...")
    run_train.main()
    print("Step 3/3: generating SHAP explanations...")
    run_explain.main()
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
