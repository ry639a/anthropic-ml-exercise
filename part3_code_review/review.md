
Overall:
Fixed indentation issues.

Transformer():
1. forward() is defined inside __init__()
2. Transformer input shape is wrong. 
    nn.TransformerEncoder expects (seq_len, batch_size, d_model) 
    Embeddings output (batch_size, seq_len, d_model). hence we need to transpose.
    x = x.transpose(0, 1)
3. There is no positional encoding. Added max_len parameter to __init__ method. and added positional encoding layer.
4. Separated initialization of EncoderLayer and TransformerEncoder.
5. Added mask to transformer forward and fixed output shape.


Training Loop:
1. Indentation is broken. Move forward/backward inside loop.
2. optimizer.zero_grad() is not present. This fails to zero out gradients and accumulates gradients over all iterations.
3. There is no model.train()
4. Optimizer Adam is too high for transformers. Use weight decay and reduced value.
5. No device handling for GPU.
6. DataLoader handling is wrong with batch.
7. Corrected total loss initialization and added correct update logic.
8. Corrected output print statement.