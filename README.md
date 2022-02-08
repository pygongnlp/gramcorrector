# gramcorrector
A simple tutoral for error correction task, based on Pytorch

### Corpus
#### SIGHAN
using SIGHAN13, 14, 15 & Augumented 28W sentences
preprocess following <https://github.com/gitabtion/BertBasedCorrectionModels/>     

|  Train  |  Valid  |  Test  |  
|  ----  | ----  |  ----  |
| 251841/2921  | 27982/293 | 1100/558  |  


**Grammatical Error Detection (sentence-level)**

a binary sequence-based classification task, use to predict if one sentence is grammatical.

**Grammatical Error Detection (token-level)**  
a binary token-based classification task, a coarse version of error detection, not only
predict if one sentence is grammatical, also output which token is wrong in an error sentence.

**Spell Error Correction (Chinese)**  
use to correct spell error in chinese

**Grammatical Error Correction**  
use to correct all type of errors


### Connect  
Peiyuan Gong(BIT)  pygongnlp@gmail.com



