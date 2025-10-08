# BBC News Topic Modeling

A topic modeling project that discovers and identifies topics in BBC news articles using Latent Dirichlet Allocation (LDA).

## Project Overview
This project implements unsupervised topic modeling on the BBC news dataset using:

- **Advanced text preprocessing** with custom regex patterns, POS tagging, and lemmatization
- **Latent Dirichlet Allocation (LDA)** with optimized hyperparameters for topic discovery
- **Parallel processing architecture** for efficient document preprocessing and model training
- **Bag-of-Words vectorization** with vocabulary filtering and extremes handling
- **Interactive web interface** built with Streamlit for real-time topic prediction
- **Modular, production-ready code** with YAML configurations and comprehensive logging
- **Complete inference pipeline** supporting both batch and single document prediction

## Topics Identified
- 🏛️ **Politics** (Topic 0): election, party, minister, tax, law, plan, issue, country, leader, case
- 💼 **Business** (Topic 1): market, firm, sale, price, growth, rate, share, economy, business, deal  
- 🎬 **Entertainment** (Topic 2): film, music, show, award, song, TV, star, director, band, movie
- 💻 **Technology** (Topic 3): technology, phone, service, computer, user, network, firm, software, system, site
- ⚽ **Sports** (Topic 4): game, player, team, match, side, club, title, minute, injury, season

## Features
- **Intelligent preprocessing pipeline** with POS-tag filtering (nouns only) and WordNet lemmatization
- **Efficient parallel processing** utilizing all CPU cores for scalable document preprocessing
- **Interactive Streamlit web app** with real-time topic prediction and probability visualization
- **Robust vocabulary filtering** reducing 10,014 initial terms to 837 meaningful words
- **Model persistence** with Gensim serialization for production deployment
- **Comprehensive logging system** with step-by-step pipeline progress tracking
- **Flexible YAML configuration** enabling hyperparameter tuning without code changes
- **Cross-platform compatibility** with automated directory creation and path handling

## Installation
### Prerequisites
- Python 3.8+
- conda or pip

### Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/ahmedxnov/topic-modeling-bbc
   cd topic-modeling-bbc
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required NLTK data:**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('omw-1.4')"
   ```

## Project Structure

```
topic-modeling-bbc/
├── config/                    # Configuration files
│   └── config.yaml           # LDA model hyperparameters
├── dataset/                   # Dataset directory
│   └── Labeled BBC.csv       # BBC news articles dataset
├── models/                    # Trained models (generated after training)
│   ├── lda_model.gensim      # Trained LDA model
│   ├── lda_model.gensim.expElogbeta.npy
│   ├── lda_model.gensim.id2word
│   ├── lda_model.gensim.state
│   └── vocabulary.dict       # Vocabulary dictionary
├── scripts/                   # Executable scripts
│   ├── app.py                # Streamlit web interface
│   ├── inference.py          # Topic prediction for new documents
│   └── train.py              # Main training pipeline script
├── src/                       # Source code modules
│   ├── data/                 # Data processing modules
│   │   ├── __init__.py
│   │   └── dataset_pipeline.py  # Data loading and parallel processing
│   ├── utils/                # Utility modules
│   │   ├── __init__.py
│   │   ├── constants.py      # Text preprocessing patterns and NLP tools
│   │   └── preprocessing.py  # Text preprocessing pipeline with POS filtering
│   └── __init__.py
├── .gitignore                # Git ignore file
├── LICENSE                   # MIT License
├── README.md                 # Project documentation
└── requirements.txt          # Project dependencies
```

## Usage

### Training
```bash
python scripts/train.py
```

### Inference
```bash
python scripts/inference.py --new_data "your_document.txt"
```

### Web App
```bash
streamlit run scripts/app.py
```

Then open your browser to `http://localhost:8501`

## Dataset
- **Source**: [BBC Dataset](http://mlg.ucd.ie/datasets/bbc.html)
- **Documents**: 2,127 unique articles (duplicates removed)
- **Original size**: 2,225 documents
- **Categories**: Business, Entertainment, Politics, Sport, Tech

## Model Details
- **Algorithm**: Latent Dirichlet Allocation (LDA)
- **Implementation**: Gensim
- **Representation**: Bag of Words
- **Topics**: 5
- **Vocabulary**: 837 words (after filtering extremes)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/amazing-feat`)
3. Commit your changes (`git commit -m 'Add amazing feat'`)
4. Push to the branch (`git push origin feat/amazing-feat`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.