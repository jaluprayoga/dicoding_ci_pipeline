import os
import urllib.request
from dotenv import load_dotenv

# Load environment variables (override existing ones)
load_dotenv(override=True)

def main():

    base_url = os.getenv("CSV_URL")
    files = [
        "X_train.csv", "y_train.csv",
        "X_train_smote.csv", "y_train_smote.csv",
        "X_test.csv", "y_test.csv"
    ]
    
    # Determine the destination storage directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for file_name in files:
        url = f"{base_url}/{file_name}"
        dest_path = os.path.join(script_dir, file_name)
        
        print(f"Downloading {file_name}...")
        try:
            # Download the file directly
            urllib.request.urlretrieve(url, dest_path)
            print(f"SUCCESS: Downloaded {file_name}")
        except Exception as e:
            print(f"ERROR: Failed to download {file_name}: {e}")
            
    print("\nAll datasets downloaded successfully!")

if __name__ == "__main__":
    main()
