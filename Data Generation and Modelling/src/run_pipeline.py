from dataset_builder import generate_dataset
from train_models import train_models

def main():
    df = generate_dataset(1000)
    train_models(df)
    
if __name__ == "__main__":
    main()