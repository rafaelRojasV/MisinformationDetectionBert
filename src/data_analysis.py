import pandas as pd


def final_check(csv_path):
    print(f"[INFO] Loading {csv_path} for final check...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Shape: {df.shape}")
    print("Columns:", list(df.columns))

    # 1) Missing values
    missing_counts = df.isnull().sum()
    print("\n=== Missing Values ===")
    print(missing_counts)

    # 2) Ensure 'combined_text' + 'label' exist
    required_cols = ['combined_text', 'label']
    for col in required_cols:
        if col not in df.columns:
            print(f"[WARNING] Column '{col}' not found—cannot proceed with typical text classification.")

    # 3) Text Length check (post-truncation)
    if 'combined_text' in df.columns:
        df['text_len'] = df['combined_text'].astype(str).apply(lambda x: len(x.split()))
        max_len = df['text_len'].max()
        print(f"\n=== Text Length after truncation ===")
        print(df['text_len'].describe())
        if max_len > 512:
            print(f"[WARNING] Found a row with >512 words ({max_len}). Possible truncation or chunking issue.")
    else:
        print("[WARNING] No 'combined_text' column present to check text length.")

    # 4) Label distribution
    if 'label' in df.columns:
        print("\n=== Label Distribution ===")
        print(df['label'].value_counts(dropna=False))
    else:
        print("[WARNING] No 'label' column found to check distribution.")

    # 5) Domain Prob or other columns
    if 'domain_prob' in df.columns:
        print("\n=== domain_prob Stats ===")
        print(df['domain_prob'].describe())

    # 6) Sample rows
    print("\n=== Sample Rows ===")
    print(df.head(3).to_string())

    print("\n=== Recommended Next Steps ===")
    print("- [ ] Check if your dataset is large enough and if you want to do any sub-sampling or filtering.")
    print("- [ ] Split into train/val/test sets if you haven’t already.")
    print("- [ ] Confirm your label 0/1 logic (fake vs. true).")
    print("- [ ] Possibly remove or ignore columns with high missingness unless they’re critical.")
    print("- [ ] You’re now ready to move on to tokenization & model fine-tuning!\n")


if __name__ == "__main__":
    csv_path = r"C:\Users\Rafael\PycharmProjects\MisinformationDetectionBert\data\processed\pure_political_dataset_truncated.csv"
    final_check(csv_path)
