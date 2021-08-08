## IATN
Dataset and source code for our paper: **[Inateraction Attention Transfer Network for Cross-domain Sentiment Classification](http://staff.ustc.edu.cn/~cheneh/paper_pdf/2019/Kai-Zhang-AAAI.pdf).**

### Amazon Review Dataset
The public dataset has been uploaded.

### Crowdfunding project Dataset: [Indiegogo.com](https://www.indiegogo.com/)
This is our private dataset, if you want to use it, please indicate the source, thank you！


### Requirements
— Python 2.7.5

—Tensorflow-gpu 1.2.1

— Numpy 1.13.3

— [Google Word2Vec](https://code.google.com/archive/p/word2vec/) 

— sklearn

— other pakages

—To install requirements, please run `pip install -r requirements.txt.`

### Environment

— OS: CentOS Linux release 7.7.1908

— CPU: 24 E5-2650 v4 @ 2.20GHz

— GPU: 4 * K80:11441 MB

### Running

**Prepare the Pre-trained Word2vec :**

— 1. Get the pre-trained model and generate the embeddings ;

​               — [Google Word2Vec](https://code.google.com/archive/p/word2vec/) ;

​               — [GloVe](https://nlp.stanford.edu/projects/glove/) ;

— 2. Put the pre-trained word_embedding (Google-Word2Vec/Glove) to the coresponseding path ;

**Prepare the aspect sequence :**

— python aspect_extraction.py

*(Input/output path can be changed inner the file!)*

**Run the model :** 

 —  python train.py

*(Default dataset is Laptop; The parameters can be changed in the train.py file! (line 15~line 31))*

### Contact

If you have any problem about this library, please send us an Email at:

— [kkzhang0808@gmail.com](kkzhang0808@gmail.com)

— [kkzhang0808@mail.ustc.edu.cn](sa517494@mail.ustc.edu.cn)

### Citation

If the data and code are useful for your research, please be kindly to give us stars and cite our paper as follows:


```
@article{zhang2019interactive,\
  title={Interactive Attention Transfer Network for Cross-domain Sentiment Classification},\
  author={Zhang, Kai and Zhang, Hefu and Liu, Qi and Zhao, Hongke and Zhu, Hengshu and Chen, Enhong},\
  year={2019}\
}
```
