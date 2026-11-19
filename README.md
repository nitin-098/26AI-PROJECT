# Smart Spend Interface

A Streamlit web interface trained on the supplied Smart Spend CSV.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run with Google Colab
Colab can train a model, but Streamlit is most conveniently run locally or deployed using Streamlit Community Cloud. Upload this folder to a GitHub repository and deploy `app.py` at https://share.streamlit.io/.

## Important
The app currently trains its own Random Forest model from the uploaded CSV. To use the **exact model in your Colab notebook**, replace `train_model()` with loading the notebook's saved model and its preprocessing pipeline. Predictions may differ from your notebook until this is done.
