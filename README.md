# IDGenRec
## Overview
PyTorch implementation of the paper "IDGen: Towards LLM-RecSys Alignment with Textual ID Learning".

![](pic/overview.png)

To better align Large Language Models (LLMs) with recommendation needs, we propose representing each item as a unique, concise, semantically rich, platform-agnostic textual ID using human language tokens.

### ID Generation Example

<div align="center">
    <img src="pic/id_generator.png" width="70%" />
</div>

### Paper Link: 
[linklinklink](#)

## Requirements
See `./environment.txt`.

## Instructions
The current implementation supports only distributed training on multiple GPUs.

### For Standard Sequential Recommendation
We provide four preprocessed datasets: Amazon_Beauty, Amazon_Sports, Amazon_Toys, and Yelp, under `./rec_datasets`. The initially generated IDs (e.g., `./rec_datasets/Beauty/item_generative_index_phase_0.txt`) are also provided.

To train the model:
1. Navigate to the command folder:
    ```
    cd command
    ```
2. Run the training script:
    ```
    sh train_standard.sh
    ```
3. You can change the `--dataset` to your desired dataset name.

### For Foundational Training
Please download the fusion dataset from [Google Drive Link](https://drive.google.com/drive/folders/1JedaL2GOxBarQTYQwKoBne_dPjDRPc52?usp=sharing) and place it under `./rec_datasets`, which mixes selected Amazon datasets. Please refer to the paper for dataset details.

To train the foundation model, run:

    ```
    sh train_foundation.sh
    ```

## Reference

```
@inproceedings{tan2024towards,
  title={IDGen: Towards LLM-RecSys Alignment based on Textual ID Learning},
  author={Juntao Tan and Shuyuan Xu and Wenyue Hua and Yingqiang Ge and Zelong Li and Yongfeng Zhang},
  booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)},
  year={2024}
}
```
