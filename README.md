# deeplearning_cw
A deep learning coursework implementing an audio-text transformer

print("Number of audio IDs:", len(val_audio))
print("Number of text IDs:", len(val_text))
print("Number of label IDs:", len(val_labels))

print("\nExample audio IDs:", list(val_audio.keys())[:5])
print("Example text IDs:", list(val_text.keys())[:5])
print("Example labels:")

for key in list(val_labels.keys())[:10]:
    print(key, val_labels[key])
