# Compressing Word Embeddings via Deep Compositional Code Learning (ICLR 2018)
PyTorch implementation and Keras for testing


![Architecture](https://github.com/msobroza/compositional_code_learning/blob/master/compositional_image.png)

I got the comparable results than the paper for sentiment analysis in the best configuration. I did not test it for Machine Translation.

https://openreview.net/forum?id=BJRZzFlRb

# Dependencies
* Keras (for testing in the LSTM IMDB sentiment analysis classification)
* tensorflow (for testing in the LSTM IMDB sentiment analysis classification)
* PyTorch
* tqdm
* torchwordemb
* numpy
* Pre-trained GloVe vectors (Download glove.42B.300d.zip from https://nlp.stanford.edu/projects/glove/)
* git
* unzip

# Execution
```bash
git clone <this_project>
cd compositional_code_learning
wget http://nlp.stanford.edu/data/glove.42B.300d.zip
# Install all dependencies
unzip glove.42B.300d.zip
# The follow line generates a dataset containing only words and vectors found in IMDB and in GloVe
python gen_intersect_imdb_embeddings.py
# Learn the compact representation (please consult help for more options)
python gumbel_softmax_ae.py --path_output_codes <path> --path_output_reconstruction <path> --version <version_name>
# Test vectors using a LSTM Model for IMDB Sentiment Analysis Classification
python lstm_sent.py
```

If you liked please put a star little star for me :-)
Any concerns or suggestions please contact me

Credits for the implementation: Max Raphael Sobroza Marques
Thanks you Raphael Shu for answer some questions about the paper 
