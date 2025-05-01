# CLIP Video Embeddings Generator

This project generates CLIP embeddings for both video frames and captions. It uses OpenAI's CLIP model to create visual and textual embeddings that can be used for various applications like similarity search, recommendation systems, etc.

## Features

- Extract frames from video files
- Generate CLIP embeddings for video frames
- Process video captions and generate text embeddings
- Save embeddings for future use

## Setup and Usage

### Requirements

The script will automatically install all required dependencies:
- PyTorch
- OpenCV
- PIL
- CLIP
- Other dependencies

### Usage

Simply run the Python script:

```bash
python clip_video_embeddings.py
```

The script will:
1. Extract frames from video1.mp4 in the Data folder
2. Read the caption from video1.txt
3. Generate CLIP embeddings for both frames and caption
4. Save embeddings to the "embeddings" folder

## Output

The script will create:
- `frame_embeddings_{timestamp}.npy`: NumPy array of frame embeddings
- `caption_embeddings_{timestamp}.npy`: NumPy array of caption embeddings

These embeddings can be used for similarity search, content recommendation, and other applications.

## Customization

You can modify the script to:
- Process different videos
- Change the number of frames extracted
- Use different CLIP model variants
- Customize embedding output format 