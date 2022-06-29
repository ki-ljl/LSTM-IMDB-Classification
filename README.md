# LSTM-IMDB-Classification
Use PyTorch to build an LSTM model for text classification on the IMDB dataset.

# Environment
pytorch==1.10.1+cu111

numpy==1.18.5

pandas==1.2.3

# Usage
1. Download the glove.6B.50d.txt file from [kaggle](https://www.kaggle.com/datasets/watts2/glove6b50dtxt).
2. Generate vocabulary_vectors.npy and word_list.npy:
```python
if __name__ == '__main__':
    load_cab_vector()
```
3. Generate sentence_code_1.npy and sentence_code_2.npy:
```python
if __name__ == '__main__':
    # load_cab_vector()
    process_sentence('train')
    process_sentence('test')
```
4. Generate training and test sets:
```python
if __name__ == '__main__':
    # load_cab_vector()
    # process_sentence('train')
    # process_sentence('test')
    process_batch(batch_size=100)
```
5. Model training and testing:
```python
if __name__ == '__main__':
    train()
    test()
    # load_cab_vector()
    # process_sentence('train')
    # process_sentence('test')
    # process_batch(100)
```
