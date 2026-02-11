
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pickle

MODEL_WEIGHTS_PATH = "best_caption_model.pt"
VOCAB_PATH = "vocab.pkl"

class Encoder(nn.Module):
    def __init__(self, feature_dim=2048, hidden_size=512):
        super().__init__()
        self.fc = nn.Linear(feature_dim, hidden_size)
        self.act = nn.Tanh()

    def forward(self, features):
        h = self.act(self.fc(features))
        h = h.unsqueeze(0)
        return h


class DecoderGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_size=512, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq, hidden):
        emb = self.embedding(input_seq)
        out, hidden = self.gru(emb, hidden)
        logits = self.fc_out(out)
        return logits, hidden

    def decode_step(self, token, hidden):
        emb = self.embedding(token).unsqueeze(1)
        out, hidden = self.gru(emb, hidden)
        logits = self.fc_out(out.squeeze(1))
        return logits, hidden


class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, pad_idx, feature_dim=2048, hidden_size=512, embed_dim=256):
        super().__init__()
        self.encoder = Encoder(feature_dim=feature_dim, hidden_size=hidden_size)
        self.decoder = DecoderGRU(vocab_size=vocab_size, embed_dim=embed_dim, hidden_size=hidden_size, pad_idx=pad_idx)

    def forward(self, features, input_seq):
        hidden = self.encoder(features)
        logits, _ = self.decoder(input_seq, hidden)
        return logits


START_TOKEN = "<start>"
END_TOKEN = "<end>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def ids_to_caption(ids, idx2word):
    words = []
    for i in ids:
        w = idx2word.get(int(i), UNK_TOKEN)
        if w == END_TOKEN:
            break
        if w in [START_TOKEN, PAD_TOKEN]:
            continue
        words.append(w)
    return " ".join(words)


@torch.no_grad()
def generate_caption_greedy(model, feature_vec, word2idx, idx2word, device, max_len=40):
    model.eval()
    
    START_ID = word2idx[START_TOKEN]
    END_ID = word2idx[END_TOKEN]

    if feature_vec.dim() == 1:
        feature_vec = feature_vec.unsqueeze(0)
    feature_vec = feature_vec.to(device)

    hidden = model.encoder(feature_vec)
    token = torch.tensor([START_ID], device=device, dtype=torch.long)
    generated_ids = [START_ID]

    for _ in range(max_len):
        logits, hidden = model.decoder.decode_step(token, hidden)
        next_id = int(torch.argmax(logits, dim=-1).item())

        generated_ids.append(next_id)

        if next_id == END_ID:
            break

        token = torch.tensor([next_id], device=device, dtype=torch.long)

    return ids_to_caption(generated_ids, idx2word)


@torch.no_grad()
def generate_caption_beam(model, feature_vec, word2idx, idx2word, device, max_len=40, k=3, length_penalty=0.7):
    model.eval()
    
    START_ID = word2idx[START_TOKEN]
    END_ID = word2idx[END_TOKEN]

    if feature_vec.dim() == 1:
        feature_vec = feature_vec.unsqueeze(0)
    feature_vec = feature_vec.to(device)

    hidden0 = model.encoder(feature_vec)
    beams = [([START_ID], hidden0, 0.0, False)]

    for _ in range(max_len):
        new_beams = []

        for seq, hidden, score, ended in beams:
            if ended:
                new_beams.append((seq, hidden, score, True))
                continue

            last_token = torch.tensor([seq[-1]], device=device, dtype=torch.long)
            logits, next_hidden = model.decoder.decode_step(last_token, hidden)

            log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
            topk_log_probs, topk_ids = torch.topk(log_probs, k)

            for lp, tid in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                new_seq = seq + [tid]
                new_score = score + lp
                new_ended = (tid == END_ID)
                new_beams.append((new_seq, next_hidden, new_score, new_ended))

        def norm_score(item):
            seq, _, score, _ = item
            L = max(1, len(seq))
            return score / (L ** length_penalty)

        new_beams.sort(key=norm_score, reverse=True)
        beams = new_beams[:k]

        if all(b[3] for b in beams):
            break

    best_seq = max(beams, key=lambda x: x[2])[0]
    return ids_to_caption(best_seq, idx2word)


@st.cache_resource
def load_feature_extractor():
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor.eval()
    return feature_extractor


def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])


@torch.no_grad()
def extract_features(image, feature_extractor, transform, device):
    img_tensor = transform(image).unsqueeze(0).to(device)
    features = feature_extractor(img_tensor)
    features = features.view(1, -1)
    return features.squeeze(0)


@st.cache_resource
def load_vocabulary():
    try:
        with open(VOCAB_PATH, 'rb') as f:
            vocab_data = pickle.load(f)
        return vocab_data['word2idx'], vocab_data['idx2word']
    except FileNotFoundError:
        st.error(f"Vocabulary file not found at: {VOCAB_PATH}")
        return None, None


@st.cache_resource
def load_captioning_model(_word2idx):
    if _word2idx is None:
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vocab_size = len(_word2idx)
    PAD_IDX = _word2idx[PAD_TOKEN]
    
    model = ImageCaptioningModel(
        vocab_size=vocab_size,
        pad_idx=PAD_IDX,
        feature_dim=2048,
        hidden_size=512,
        embed_dim=256
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Model weights not found at: {MODEL_WEIGHTS_PATH}")
        return None


def main():
    st.set_page_config(
        page_title="Image Captioning",
        layout="centered"
    )
    
    st.title("Image Captioning with Neural Networks")
    st.markdown("Upload an image and the model will generate a caption for it!")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.sidebar.info(f"Running on: **{device}**")
    
    word2idx, idx2word = load_vocabulary()
    
    if word2idx is None:
        st.warning("Vocabulary not loaded. Please check the file path.")
        return
    
    model = load_captioning_model(word2idx)
    
    if model is None:
        st.warning("Model not loaded. Please check the file path.")
        return
    
    feature_extractor = load_feature_extractor().to(device)
    transform = get_image_transform()
    
    st.success("Model and vocabulary loaded successfully!")
    
    st.sidebar.header("Generation Settings")
    
    decoding_method = st.sidebar.selectbox(
        "Decoding Method",
        ["Greedy", "Beam Search", "Both"]
    )
    
    max_length = st.sidebar.slider("Max Caption Length", 10, 50, 40)
    
    if decoding_method in ["Beam Search", "Both"]:
        beam_width = st.sidebar.slider("Beam Width (k)", 2, 10, 3)
        length_penalty = st.sidebar.slider("Length Penalty", 0.0, 1.5, 0.7)
    else:
        beam_width = 3
        length_penalty = 0.7
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp']
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            with st.spinner("Generating caption..."):
                features = extract_features(image, feature_extractor, transform, device)
                
                st.subheader("Generated Captions")
                
                if decoding_method in ["Greedy", "Both"]:
                    greedy_caption = generate_caption_greedy(
                        model, features, word2idx, idx2word, device, max_len=max_length
                    )
                    st.markdown(f"**Greedy:** {greedy_caption}")
                
                if decoding_method in ["Beam Search", "Both"]:
                    beam_caption = generate_caption_beam(
                        model, features, word2idx, idx2word, device,
                        max_len=max_length, k=beam_width, length_penalty=length_penalty
                    )
                    st.markdown(f"**Beam Search (k={beam_width}):** {beam_caption}")


if __name__ == "__main__":
    main()
