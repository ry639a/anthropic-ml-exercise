
# TODO
# * README with setup instructions and how to reproduce results  * Explanation of design decisions 
# * Discussion of results and potential improvements 

# Transformer-based classifier for sentiment analysis

Setup Instructions:
1. Create and activate your virtual environment:
    python -m venv torch_venv
    For venv (Windows): .\.\torch_venv\Scripts\activate
2. Update the python interpreter for the project to be used from virtual environment.
3. Install ML libraries:
    pip install numpy datasets wandb matplotlib seaborn
4. pip freeze > requirements.txt

5. pip install -r requirements.txt
6. pip install --upgrade pip
7. pip install numpy
7. python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126


Steps:
1. tokenization: SentencePiece or BytePair encoding.
2. Create Vector Embeddings for each word.
3. Add positional encoding: Add positional vector to each word vector embedding.
4. Multi Head Attention
5. Feed Forward Network
6. Softmax