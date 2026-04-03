"""
Preprocess dataset for VAE-DiT TTS training.

Expects AISHELL-3 style directory:
  base_dir/
    train/
      wavs/
        SSB0001/
          SSB00010001.wav
          SSB00010002.wav
        SSB0002/
          ...
      content.txt   (lines like: "SSB00010001 pinyin1 汉字1 pinyin2 汉字2 ...")
    test/
      ...

Output:
  processed_dir/
    train/
      wavs/
        SSB0001_SSB00010001.pt   (VAE latent)
      content.txt                (lines like: "SSB0001_SSB00010001_汉字汉字汉字")
    test/
      ...
"""

import os
import argparse
import torch
import torchaudio
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import io
# Add project root to sys.path to allow importing from models
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vae import load_vae, vae_encode


def handle_wav(base_dir: str, processed_dir: str, split: str, vae):
    """Encode all wavs to VAE latents."""
    path_wav = os.path.join(base_dir, split, 'wav')
    out_wav_dir = os.path.join(processed_dir, split, 'wav')
    os.makedirs(out_wav_dir, exist_ok=True)

    folder = Path(path_wav)
    file_paths = [str(p) for p in folder.rglob('*.wav')]
    print(f"Found {len(file_paths)} wav files in {path_wav}")

    with torch.no_grad():
        for i in tqdm(file_paths, desc=f"Encoding {split}"):
            file = Path(i)
            wav, sr = torchaudio.load(file)

            # # Convert to mono
            # if wav.shape[0] > 1:
            #     wav = wav.mean(dim=0, keepdim=True)

            # Resample to 48kHz
            if sr != 48000:
                wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=48000)
            wav = torch.clamp(wav, -1.0, 1.0).to(device=vae.device, dtype=vae.dtype).unsqueeze(1).repeat(1, 2, 1)
            # print(f'wav_shape!:{wav.shape}')
            # VAE encode: (1, 1, samples) → (1, T, D)
            latent = vae_encode(vae, wav)
            # print(f'latent_shape:{latent.shape}')
            latent = latent.squeeze(0).cpu()  # (T, D)

            # Save: {speaker}_{utterance_id}.pt
            out_name = f"{file.parent.name}_{file.stem}.pt"
            torch.save(latent, os.path.join(out_wav_dir, out_name))

def handle_jvs_audio_and_text(base_dir, processed, vae):
    """
    jvs/jvsxxx/falsetxx/*wav
    jvs/jvsxxx/falsetxx/*txt
    """
    path = Path(base_dir)
    text_line = []
    if not os.path.exists(processed):
      os.makedirs(processed)
    for child in tqdm(list(path.iterdir()), desc="Processing JVS dataset"):
      if not child.is_dir():
        continue
      for child_1 in ['parallel100', 'nonpara30', 'whisper10']:
          audio_path = child / child_1 / 'wav24kHz16bit'
          files = [f for f in audio_path.iterdir() if f.is_file()]
          for file in files:
              if file.suffix == '.wav':
                  wav, sr = torchaudio.load(file)
                  if sr != 48000:
                      wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=48000)
                  wav = torch.clamp(wav, -1.0, 1.0).to(device=vae.device, dtype=vae.dtype).unsqueeze(1).repeat(1, 2, 1)
                  latent = vae_encode(vae, wav)
                  latent = latent.squeeze(0).cpu()  # (T, D)
                  if child_1 != 'whisper10':
                    out_name = f"{child.name}-normal-{child.name +file.stem.replace('_', '-')}.pt"
                  else:
                    out_name = f"{child.name}-whisper-{child.name +file.stem.replace('_', '-')}.pt"
                  torch.save(latent, os.path.join(processed, out_name))
          with open(os.path.join(child, child_1, 'transcripts_utf8.txt'), 'r', encoding='utf-8') as f:
            for line in f:
              line = line.strip()
              if not line:
                continue
              parts = line.split(':')
              file_name = parts[0]  # e.g. *.wav
              speaker = child.name  # e.g. jvs001
              only_text = parts[1]
              if child_1 != 'whisper10':
                out_name = f"{speaker}-normal_{child.name}-normal-{child.name +file_name.replace('_', '-')}.pt" #speaker_file_name_content to match previous dataset
              else:
                out_name = f"{speaker}-whisper_{child.name}-whisper-{child.name +file_name.replace('_', '-')}.pt"
              text_line.append(f"{out_name}_{only_text}\n")
    with open(os.path.join(processed, 'content.txt'), 'w', encoding='utf-8') as fout:
      fout.writelines(text_line)
                

def handle_LibriTTS_audio_and_text(base_dir, processed, vae):
    """
    LibriTTS/train/speaker/char./wav txt
    """
    path = Path(base_dir)
    text_line = []
    if not os.path.exists(processed):
      os.makedirs(processed)
    skipped = 0
    count = 0
    for child in tqdm(list(path.iterdir()), desc="Processing LibriTTS dataset"):
      if not child.is_dir():
        continue
      for child_1 in child.iterdir():
          if not child_1.is_dir():
            continue
          files = [f for f in child_1.iterdir() if f.is_file() and f.suffix == '.wav']
          for file in files:
            save_file_stem = file.stem.replace('_', '-')
            out_pt = os.path.join(processed, save_file_stem + '.pt')
            speaker = child.name

            # Skip if already encoded
            if os.path.exists(out_pt):
              # Still need text for content.txt
              txt_path = os.path.join(child_1, f'{file.stem}.normalized.txt')
              if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                  text = f.read().strip()
                text_line.append(f"{speaker}_{save_file_stem}.pt_{text}\n")
              continue

            # Read text — skip if missing
            txt_path = os.path.join(child_1, f'{file.stem}.normalized.txt')
            if not os.path.exists(txt_path):
              skipped += 1
              continue
            with open(txt_path, 'r', encoding='utf-8') as f:
              text = f.read().strip()

            try:
              wav, sr = torchaudio.load(file)
              if sr != 48000:
                  wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=48000)
              wav = torch.clamp(wav, -1.0, 1.0).to(device=vae.device, dtype=vae.dtype).unsqueeze(1).repeat(1, 2, 1)
              latent = vae_encode(vae, wav)
              latent = latent.squeeze(0).cpu()  # (T, D)
              torch.save(latent, out_pt)
              text_line.append(f"{speaker}_{save_file_stem}.pt_{text}\n")
              count += 1
            except Exception as e:
              print(f"\n  Skipping {file.name}: {e}")
              skipped += 1
            # finally:
            #   # Free GPU memory
            #   for v in ['wav', 'latent']:
            #     if v in locals():
            #       del locals()[v]
            #   if count % 100 == 0:
            #     torch.cuda.empty_cache()

    print(f"Processed {count}, skipped {skipped}")
    with open(os.path.join(processed, 'content.txt'), 'w', encoding='utf-8') as fout:
      fout.writelines(text_line)


def handle_FGO_audio_and_text(base_dir, processed, vae):
  import pandas as pd
  import re
  if not os.path.exists(processed):
    os.makedirs(processed)
  text_line = []
  df = pd.read_parquet(os.path.join(base_dir, 'table.parquet'))
  skipped = 0
  for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing FGO dataset"):
    raw_filename = row['filename']
    safe_stem = re.sub(r'[*?<>|":\\]', '', raw_filename).replace('_', '-')
    out_pt = os.path.join(processed, safe_stem + '.pt')

    if os.path.exists(out_pt):
      text = row['voice_text']
      speaker = str(row['char_name'])
      text_line.append(f"{speaker}_{safe_stem}.pt_{text}\n")
      continue

    audio_path = os.path.join(base_dir, raw_filename)
    if not os.path.exists(audio_path):
      skipped += 1
      continue

    try:
      wav, sr = torchaudio.load(audio_path)
      if sr != 48000:
          wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=48000)
      wav = torch.clamp(wav, -1.0, 1.0).to(device=vae.device, dtype=vae.dtype).unsqueeze(1).repeat(1, 2, 1)
      latent = vae_encode(vae, wav)
      latent = latent.squeeze(0).cpu()  # (T, D)
      torch.save(latent, out_pt)

      text = row['voice_text']
      speaker = str(row['char_name'])
      text_line.append(f"{speaker}_{safe_stem}.pt_{text}\n")
    except Exception as e:
      print(f"\n  Skipping {raw_filename}: {e}")
      skipped += 1
    # finally:
    #   # Free GPU memory every iteration
    #   del wav, latent
    #   if index % 100 == 0:
    #     torch.cuda.empty_cache()

  print(f"Skipped {skipped} files")
  with open(os.path.join(processed, 'content.txt'), 'w', encoding='utf-8') as fout:
    fout.writelines(text_line)

def handle_asmr_text(base_dir, processed_dir, vae):
    file_lists = os.listdir(base_dir)
    text_line = []
    flac_files = [f for f in file_lists if f.endswith('.flac')]
    for file in tqdm(flac_files, desc="Processing ASMR audio"):
        audio_path = os.path.join(base_dir, file)
        try:
            wav, sr = torchaudio.load(audio_path)
            if sr != 48000:
                wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=48000)
            wav = torch.clamp(wav, -1.0, 1.0).to(device=vae.device, dtype=vae.dtype)
            # (C, samples): mono → duplicate to stereo, stereo → keep
            if wav.shape[0] == 1:
                print('[DEBUG] Mono audio detected, duplicating to stereo')
                wav = wav.repeat(2, 1)  # (1, S) → (2, S)
            latent = vae_encode(vae, wav.unsqueeze(0))  # (1, 2, S) → (1, T, D)
            latent = latent.squeeze(0).cpu()  # (T, D)
            torch.save(latent, os.path.join(processed_dir, file.split('.')[0] + '.pt'))

            text_file = file.split('.')[0] + '.txt'
            with open(os.path.join(base_dir, text_file), 'r', encoding='utf-8') as f:
                text = f.read().strip()
            text_line.append(f"{file.split('.')[0]}.pt_{text}\n")
        except Exception as e:
            print(f"Skipping {file}: {e}")

    with open(os.path.join(processed_dir, 'content.txt'), 'w', encoding='utf-8') as fout:
        fout.writelines(text_line)
        
def handle_Japanese_Eroge(base_dir, processed_dir, vae):
    files_tables = os.listdir(base_dir)
    filtered_tables = [f for f in files_tables if f.endswith(".parquet")]
    text_lines = []
    prefix_count = 0
    skipped_count = 0
    processed_count = 0
    wav_dir = os.path.join(processed_dir, "wav")
    os.makedirs(wav_dir, exist_ok=True)

    for file in tqdm(filtered_tables, desc="Processing Japanese Eroge parquet"):
        base_path = os.path.join(base_dir, file)
        try:
            df = pd.read_parquet(base_path)
        except Exception as e:
            print(f"Skipping parquet {file}: {e}")
            continue

        for index, row in tqdm(df.iterrows(), total=len(df), desc=file, leave=False):
            try:
                audio_info = row["audio"]
                audio_bytes = audio_info["bytes"] if isinstance(audio_info, dict) else audio_info
                file_obj = io.BytesIO(audio_bytes)
                wav, sample_rate = torchaudio.load(file_obj)
                if sample_rate != 48000:
                    wav = torchaudio.functional.resample(wav, orig_freq=sample_rate, new_freq=48000)
                wav = torch.clamp(wav, -1.0, 1.0).to(device=vae.device, dtype=vae.dtype)
                if wav.shape[0] == 1:
                    wav = wav.repeat(2, 1)
                latent = vae_encode(vae, wav.unsqueeze(0))
                latent = latent.squeeze(0).cpu()

                output_file_stem = f"eroge-{prefix_count}-{index}"
                output_file_name = f"{output_file_stem}.pt"
                prefix_count += 1
                torch.save(latent, os.path.join(wav_dir, output_file_name))
                text_lines.append(f"none_{output_file_name}_{row['text']}\n")
                processed_count += 1
            except Exception as e:
                skipped_count += 1
                tqdm.write(f"Skipping row {index} in {file}: {e}")

    with open(os.path.join(processed_dir, "content.txt"), "w", encoding="utf-8") as file:
        file.writelines(text_lines)
    print(f"Japanese Eroge preprocessing complete: processed={processed_count}, skipped={skipped_count}")

def handle_txt(base_dir: str, processed_dir: str, split: str):
    """
    Parse content.txt and extract Chinese characters only.

    AISHELL-3 format: "SSB00010001 pinyin1 汉字1 pinyin2 汉字2 ..."
    Output format:    "SSB0001_SSB00010001_汉字1汉字2..."
    """
    path_txt = os.path.join(base_dir, split, 'content.txt')
    out_txt = os.path.join(processed_dir, split, 'content.txt')
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)

    with open(path_txt, 'r', encoding='utf-8') as fin, \
         open(out_txt, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split( )
            utterance_id = parts[0]  # e.g. SSB00010001
            speaker = utterance_id[:7]  # e.g. SSB0001
            # Extract Chinese chars (every other token after the utterance ID)
            # AISHELL-3: "SSB00010001 pin1 汉 pin2 字 ..."
            only_text = ''.join(parts[1::2])  # skip id and pinyins
            fout.write(f"{speaker}_{utterance_id}_{only_text}\n")

    print(f"Processed text for {split}: {out_txt}")


with torch.no_grad():
  def main():
      parser = argparse.ArgumentParser(description="Preprocess dataset for VAE-DiT TTS")
      parser.add_argument("--base_dir", type=str, required=True, help="Path to raw dataset (AISHELL-3)")
      parser.add_argument("--processed_dir", type=str, default=None, help="Output directory")
      parser.add_argument("--vae_path", type=str, default="models/vae_model", help="Path to VAE model")
      parser.add_argument("--splits", nargs="+", default=["train", "test"])
      parser.add_argument("--dataset_name", type=str, default="AISHELL-3", help="dataset name")
      args = parser.parse_args()

      if args.processed_dir is None:
          args.processed_dir = os.path.join(args.base_dir, 'processed')

      # Process text first (fast)
      if args.dataset_name == "AISHELL-3":
        for split in args.splits:
          handle_txt(args.base_dir, args.processed_dir, split)
  
      # Load VAE and encode audio
      device = "cuda" if torch.cuda.is_available() else "cpu"
      vae = load_vae(args.vae_path, device=device,precision='fp16')
      if args.dataset_name == "AISHELL-3":
        for split in args.splits:
            handle_wav(args.base_dir, args.processed_dir, split, vae)
      elif args.dataset_name == "jvs":
        handle_jvs_audio_and_text(args.base_dir, args.processed_dir, vae)
      elif args.dataset_name == "LibriTTS":
        handle_LibriTTS_audio_and_text(args.base_dir, args.processed_dir, vae)
      elif args.dataset_name == "FGO":
        handle_FGO_audio_and_text(args.base_dir, args.processed_dir, vae)
      elif args.dataset_name == "asmr":
        handle_asmr_text(args.base_dir, args.processed_dir, vae)
      elif args.dataset_name == 'eroge':
         handle_Japanese_Eroge(args.base_dir, args.processed_dir, vae)
      print("Done!")


if __name__ == "__main__":
    main()
