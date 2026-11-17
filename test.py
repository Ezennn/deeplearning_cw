
import torch

for filename in [
    "train_audio_features.pt",
    "train_text_features.pt",
    "train_labels.pt"
]:
    data = torch.load(
        filename,
        map_location="cpu",
        weights_only=False
    )

    print("\n", filename)
    print("Number of entries:", len(data))

    key = next(iter(data))
    print("Example key:", key)

    value = data[key]

    if isinstance(value, torch.Tensor):
        print("Shape:", value.shape)
        print("Dtype:", value.dtype)
    else:
        print("Type:", type(value))
        print("Value:", value)
