# deeplearning_cw
A deep learning coursework implementing an audio-text transformer

Partly. The Trainer loop, argparse and logging carry over. The data, head, loss and evaluation need rewriting, and a few things in it are wrong for this task.

Task changes
1. Loss and head: this is binary matching, but fc2 outputs 10 classes with CrossEntropy and argmax. Use one logit with BCEWithLogitsLoss and threshold the sigmoid at 0.5 for accuracy.
2. Data: delete the torchvision/CIFAR/patchify/Unfold code and write a Dataset over the .pt dicts. Load one file in a REPL first and print the keys, shapes and the val/test pair structure, since I don't know the real format.
3. Negative sampling: I can't see Figure 1, so check it. If training pairs are purely random n x n, positives are about 1/n of samples. Unless the diagram says otherwise, I'd sample 50/50 to match the balanced eval, and make sure a sampled negative isn't the true match.
4. Tokens: instead of patches you have one projected audio feature plus 40 text tokens through the shared text projection. Size the positional embedding for that. I can't see Figure 2 either, so check whether it uses a CLS token or pools.

Bugs and defaults
- n_layers=2, but the spec says a single transformer layer. Set it to 1.
- self.encoder is created and never used, but it inflates your parameter count. Delete it.
- nn.TransformerEncoderLayer defaults to ReLU, dropout 0.1 and dim_feedforward 2048. The spec says GELU, and the MLP size is yours to tune. fc1 also uses ReLU. The spec's GELU/LayerNorm/dropout sentence is ambiguous for the classification head, so work out your reading and be consistent.
- lr=1e-2 with Adam is too high for a transformer. Start around 1e-4 to 1e-3.
- pos_embedding uses randn (std 1). Use something like 0.02.
- log_metrics writes "time/data" twice, so the step time overwrites the data time.
- The Trainer calls its loader "val", and main passes the test set into it. Select models on val and touch test once at the end, because that's an easy way to leak.

Missing
- Best-val checkpointing, fixed seeds, and a separate script that loads the weights and reports test accuracy deterministically. The submission requires this.
- A definition of an epoch under random pairing, e.g. N sampled pairs per epoch.
- If the features fit in GPU memory, skip DataLoader workers and batch on the GPU. It makes your iteration loop much faster.

One decision: your Attention class is unused. I'd use it instead of nn.TransformerEncoderLayer. A hand-written layer removes any argument about whether you implemented the architecture, since the brief gives 0 if it's missing. It also gives you attention weights for the qualitative section and for extensions, and full control over dropout. It's correct as far as it goes, but you'd add dropout, the residuals, LayerNorm and the MLP block.

If you paste the shapes and key structure from the feature files, I can write the Dataset and model skeleton. Since code help is a listed task, declare it in your AI usage statement.
