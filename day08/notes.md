### Lightning Attention
Lightning Attention is a faster way to compute attention in neural networks. It works by processing smaller chunks of input data (queries, keys, and values) and storing temporary results in on-chip memory, reducing the need to access slower global memory. This makes attention computation more efficient, especially for large models.

### Paper Notes
[Link to Paper:](https://arxiv.org/abs/2501.08313)
* in the lightning attention forward pass we can see that there is a mat mul part and we can bind the O_intra with a softmax 
* so we have to make a mat mul kernel and softmax kernel & then we will make a lightning attention function to write the forward pass
* i will be going for the naive mat mul kernel & naive softmax kernel
* i will be using my previous implementations for both

![alt text](algorithm1.png)
