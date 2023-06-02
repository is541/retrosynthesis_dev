import pandas as pd

if __name__ == "__main__":
    # Set the random seed for reproducibility
    seed = 42

    # Read the original CSV file
    df = pd.read_csv('guacamol_v1_test.csv')

    # Perform random subsampling
    subsampled_df = df.sample(n=10000, random_state=seed)

    # Write the subsampled data to a new CSV file
    subsampled_df.to_csv('guacamol_v1_test_10ksample.txt', sep='\n', index=False, header=False)