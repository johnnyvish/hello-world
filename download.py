# This file runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model
from audiocraft.models import AudioGen

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    AudioGen.get_pretrained('facebook/audiogen-medium')

if __name__ == "__main__":
    download_model()