
COMMANDS REFERENCE — EXACT COMMANDS TO RUN
===========================================

KAGGLE SETUP (2 cells):
  Cell 1:
    !rm -rf /kaggle/working/energivanu /kaggle/working/energivanu_prod
    !git clone https://github.com/mysterious75/Energivanu.git /kaggle/working/energivanu
    import sys; sys.path.insert(0, "/kaggle/working/energivanu")
    !pip install -q pyarrow tqdm

  Cell 2:
    !cd /kaggle/working/energivanu && python kaggle_run.py

COLAB SETUP (if using Colab instead):
  Cell 1:
    from google.colab import drive; drive.mount('/content/drive')
    !git clone https://github.com/mysterious75/Energivanu.git /content/Energivanu
    import sys; sys.path.insert(0, "/content/Energivanu")
    !pip install -q torch numpy pandas scikit-learn pyarrow tqdm matplotlib

  Cell 2:
    cd /content/Energivanu && python kaggle_run.py

HF_TOKEN SETUP (for model persistence):
  1. Go to huggingface.co/settings/tokens
  2. Create "Write" token
  3. Run: !cd /kaggle/working/energivanu && HF_TOKEN="hf_xxxxx" python kaggle_run.py

LOCAL LAPTOP CHECK:
  nvidia-smi                     # GPU status
  python -c "import torch; print(torch.cuda.is_available())"
  wmic memorychip get capacity   # RAM check

GIT COMMANDS:
  git add -A && git commit -m "message"
  git push
  git pull

VIEW CHECKPOINTS:
  !ls -la /kaggle/working/energivanu_prod/checkpoints/

VIEW HISTORY:
  !cat /kaggle/working/energivanu_prod/checkpoints/history.json

DELETE OLD DATA:
  !rm -rf /kaggle/working/energivanu_prod
