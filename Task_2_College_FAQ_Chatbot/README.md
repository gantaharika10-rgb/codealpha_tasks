# Cloud Computing FAQ Chatbot (Nimbus)

An NLP chatbot that answers cloud computing questions (service models, AWS/Azure/GCP,
Docker, security, costs, careers).

## Features
- 45 FAQs in 5 categories (edit the `FAQS` list in `chatbot.py` to add more)
- Preprocessing with NLTK: tokenize, stop-word removal, stemming
- TF-IDF + cosine similarity matching (scikit-learn)
- Confidence score, related-question chips, small talk
- Web chat UI (Flask) and terminal mode

## Run
```
pip install -r requirements.txt
python chatbot.py            # web UI
python chatbot.py --cli      # terminal chat
python chatbot.py --test     # self-test
```

## How it works
1. Clean and tokenize the message, remove stop words, stem
2. Convert it to a TF-IDF vector
3. Compare with every FAQ using cosine similarity
4. Show the best answer, or suggest close topics if the score is low
