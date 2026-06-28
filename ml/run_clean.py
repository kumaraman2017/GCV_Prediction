import json

import pandas as pd

from src.config import CLEAN_DATA_PATH, DATA_QUALITY_REPORT_PATH, PROCESSED_DATA_DIR, RAW_DATA_PATH
from src.data.clean import clean_pipeline


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(RAW_DATA_PATH)
    clean_df, report = clean_pipeline(raw_df)
    clean_df.to_csv(CLEAN_DATA_PATH, index=False)
    DATA_QUALITY_REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Cleaned {report['raw_row_count']} rows -> {report['rows_after_cleaning']} rows")
    print(f"Wrote {CLEAN_DATA_PATH} and {DATA_QUALITY_REPORT_PATH}")


if __name__ == "__main__":
    main()
