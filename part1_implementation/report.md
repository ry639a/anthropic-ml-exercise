
### Task: Classify IMDB review text to positive(1) or negative(0)

### * README with setup instructions and how to reproduce results  
### * Explanation of design decisions 
### * Discussion of results and potential improvements 

## Transformer-based classifier for sentiment analysis to classify IMDB reviews.

### Setup Instructions:
1. pip install -r requirements.txt
2. pip install --upgrade pip
3. pip install numpy
4. python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

### Optional if any setup modules are missing:
1. Create and activate your virtual environment:
    python -m venv torch_venv
    For venv (Windows): .\.\torch_venv\Scripts\activate
2. Update the python interpreter for the project to be used from virtual environment.
3. Install ML libraries:
    python -m pip install numpy datasets wandb matplotlib seaborn
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
5. pip freeze > requirements.txt


## Architecture and code flow:

#### 1. Dataset: Loading Data
   IMDB moview reviews dataset.
   Dataset link: https://ai.stanford.edu/~amaas/data/sentiment/
       For this project, data was downloaded from huggingface datasets (imdb).
       https://huggingface.co/datasets/stanfordnlp/imdb
       It consists of
           25k train samples
           25k test samples and
           50k unsupervised samples.
       Of 25k train samples 18000 samples are used for training set and 7000 are used in validation set.
   
#### 2. Tokenization:
   Reviews text is in language and neural networks or computer for that matter only understand numbers. Hence, preprocessing for this involves
   effectively converting the words present in the text to numerical representations. The first step in that process in tokenization which splits the text into tokens that are converted into numerical embeddings.
   These tokens can be sentences or words or even sub words.
   For this project, Bypte Pair enccoding is used.
   the tokenizer ByteLevelBPETokenizer from hugginface is trained with all the review texts from train, test and unsupervised samples.
   The trained tokenizer is later used to tokenize each text sample.
   Alternatives considered is SentencePiece which is very helpful in training text from specific domains like healthcare, legal etc.,
   Other tokenizer for sentence and word are avoided as we effectiely want to capture sub words as review language is usually loose and unofficial.
   
#### 4. Custom Dataset and DataLoader:
   Since we want to preserve the original text, labels and tokens,a custom dataset is created with all these fields. It also has max_len to truncate additional tokens or pad shortedned tokens.
   torch.DataLoader is used to create train_dataloader, test and val dataloaders.
   
#### 5. Vector Embeddings and Positional Encoding:
   Create numeric embeddings vector and add positional vector to each word vector embedding.
   Custom **Embeddings** class is created that creates vector embeddings and add positional mbeddings.
   layer_norm and droput is added to this layer.

#### 9. Multi Head Attention:
   Custom Multi head Attention class is created.

#### 11. Feed Forward Network:
   As in the original paper, Feedforward network with 2 linear layers and ReLU activation is used with Dropout.

#### 12. TransformerEncoderLayer:
   Custom TransformerEncoderLayer with following:
   MultiHeadAttention with LayerNorm.
   Feedforward network with layernorm and droput.
    
#### 14. TransformerEncoder
   Consists of num_layers(4) TransformerEncoderLayers.
   Linear layer to classify the output.

#### 15. All the architcture parameters can be conifgures from config.yaml.
