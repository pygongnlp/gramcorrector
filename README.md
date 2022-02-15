# gramcorrector

### Introduction
A simple recipe for building a grammatical error correction (GEC) system .

### Tasks

* **SGED (Sentence-Level Grammatical Error Detection)**
* **TGED (Token-Level Grammatical Error Detection)**
* **SEC (Spell Error Correction)**
* **GEC (Grammatical Error Correction)**

### Corpus

### Models

* **SGED**
  * **CNN**
  * **BiLSTM**
  * **BiLSTM+attn**
  * **BERT+FFN**
* **TGED**
  * **BiLSTM+CRF**
  * **BERT+softmax**
  * **BERT+CRF**
* **SEC**
  * **BERT+softmax**
  * **SoftmaskedBert+softmax**
  * **BERT+CRF**
* **GEC**
  * **Seq2seq**
    * **BiLSTM+attn**
    * **Transformer**
    * **Transformer+copy**
    * **BART**
  * **Edit**
    * **Gector**
    * **TtT**
    
### Requirements

* Python3
* Pytorch
* Transformers
* Fairseq

### Experiments

* **SGED**
* **TGED**
* **SEC**
* **GEC**

### Results

* **SGED**
* **TGED**
* **SEC**
* **GEC**

### References

### Connection
Peiyuan Gong (公培元）, from Beijing Institute of Technology （北京理工大学）  
Email:  pygongnlp@gmail.com  
Wechat:  gongpeiyuan1