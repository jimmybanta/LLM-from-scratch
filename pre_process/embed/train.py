''' File used for training various word embeddings. '''

from data.import_tiny import import_tiny
from data.import_small import import_full_list

from pre_process.embed.gensim_word2vec import GensimWord2Vec
from pre_process.tokenize.bpe import NaiveBPETokenizer



def train_gensim_word2vec_and_save(size, binary_save_path, table_save_path):
    
    if size == 'tiny':
        text = import_tiny()
    elif size in ['small', 'medium']:
        text = import_full_list()

    bpe = NaiveBPETokenizer(
        vocab_file=f'/Users/jimbo/Documents/coding/projects/llm-from-scratch/pre_process/tokenize/assets/{size}_vocab.json',
        lookup_table_file=f'/Users/jimbo/Documents/coding/projects/llm-from-scratch/pre_process/tokenize/assets/{size}_lookup_table.json',
    )

    tokenized = bpe.encode(text, return_integers=False)

    embedder = GensimWord2Vec(tokenized, size=512, window=5, min_count=1, workers=4)

    embedder.train()
    embedder.save(binary_save_path, file_type='binary')
    embedder.save(table_save_path, file_type='table')


if __name__ == '__main__':
     
    # train tiny
    train_gensim_word2vec_and_save(
        size='tiny',
        binary_save_path='/Users/jimbo/Documents/coding/projects/llm-from-scratch/pre_process/embed/assets/tiny_embeddings.bin',
        table_save_path='/Users/jimbo/Documents/coding/projects/llm-from-scratch/pre_process/embed/assets/tiny_embeddings.txt',
    )

    # train small
    train_gensim_word2vec_and_save(
        size='small',
        binary_save_path='/Users/jimbo/Documents/coding/projects/llm-from-scratch/pre_process/embed/assets/small_embeddings.bin',
        table_save_path='/Users/jimbo/Documents/coding/projects/llm-from-scratch/pre_process/embed/assets/small_embeddings.txt',
    )

    # train medium
    train_gensim_word2vec_and_save(
        size='medium',
        binary_save_path='/Users/jimbo/Documents/coding/projects/llm-from-scratch/pre_process/embed/assets/medium_embeddings.bin',
        table_save_path='/Users/jimbo/Documents/coding/projects/llm-from-scratch/pre_process/embed/assets/medium_embeddings.txt',
    )



