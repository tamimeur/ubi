import torch
import esm
import transformers
import pytorch_lightning as pl

def test_imports():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("ESM version:", esm.__version__)
    print("Transformers version:", transformers.__version__)
    print("PyTorch Lightning version:", pl.__version__)
    
    # Test loading ESM-2 model
    print("\nTesting ESM-2 model loading...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    print("ESM-2 model loaded successfully!")
    
if __name__ == "__main__":
    test_imports()
