# MASS-NAT

<!---
The source sentence in summarization tasks is usually long. To handle the long sequence, we use document-level corpus to extract long-continuous sequence for pre-training. The max sequence length is set as 512 for each iteration. For each sequence, we randomly mask a continuous segment for every 64-tokens at the encoder and predict it at the decoder. 
-->

Implement of MASS in NAT
## Dependency
```
pip install fairseq==0.8.0 pytorch_transformers
```

## Run
Please refer to mass_nat.sh
