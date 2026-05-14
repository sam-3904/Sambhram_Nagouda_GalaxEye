import torch
import rasterio
import argparse
from model import SiameseResUNet 
from dataloaders import GalaxEyeDataset
from torch.utils.data import DataLoader

def evaluate(data_path, weight_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = SiameseResUNet().to(device)
    checkpoint = torch.load(weight_path, map_location=device)
  
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Load Test Data
    test_ds = GalaxEyeDataset(data_path)
    loader = DataLoader(test_ds, batch_size=1)


    print("Running evaluation on test set...")
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    args = parser.parse_args()
    evaluate(args.data_path, args.weights)
