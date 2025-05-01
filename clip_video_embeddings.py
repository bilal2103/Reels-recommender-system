import os
import sys
import subprocess
import torch
import numpy as np
import cv2
from PIL import Image
import time
import clip
import glob
import argparse

def extract_frames(video_path, num_frames=None, frame_interval=1):
    """Extract frames from the video
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract. If None, extract all frames
        frame_interval: Interval between frames (e.g., 1 = every frame, 2 = every other frame)
    """
    print(f"Extracting frames from {video_path}...")
    video = cv2.VideoCapture(video_path)
    
    # Get video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f"Video has {total_frames} frames, {fps} fps, duration: {duration:.2f} seconds")
    
    frames = []
    
    # If num_frames is None or greater than total_frames, extract all frames with the given interval
    if num_frames is None or num_frames >= total_frames:
        frame_indices = range(0, total_frames, frame_interval)
    else:
        # Take evenly spaced frames
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    for idx in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = video.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    video.release()
    print(f"Extracted {len(frames)} frames")
    return frames

# Add GPU detection and info
def print_gpu_info():
    print("\nGPU Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA-capable GPU detected")
        
    # Check if MPS (Metal Performance Shaders) is available for Mac users
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("MPS (Metal Performance Shaders) is available")
            return "mps"
    except:
        pass
    
    # Determine best device
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def get_clip_embeddings(frames, model, preprocess, device, batch_size=32):
    """Generate CLIP embeddings for frames"""
    # Preprocess frames for CLIP
    preprocessed_frames = [preprocess(Image.fromarray(frame)) for frame in frames]
    preprocessed_frames_tensor = torch.stack(preprocessed_frames)
    
    # Get frame embeddings
    print("Generating frame embeddings...")
    with torch.no_grad():
        # Process in batches if there are many frames
        num_batches = (len(preprocessed_frames_tensor) + batch_size - 1) // batch_size
        
        frame_features_list = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(preprocessed_frames_tensor))
            batch = preprocessed_frames_tensor[start_idx:end_idx].to(device)
            
            batch_features = model.encode_image(batch)
            frame_features_list.append(batch_features.cpu().numpy())
        
        # Concatenate all batches
        frame_features = np.vstack(frame_features_list)
    
    return frame_features

def split_text_into_chunks(text, max_tokens=77):
    """Split text into chunks that fit within CLIP's token limit.
    
    Args:
        text: Text to split
        max_tokens: Maximum number of tokens per chunk (CLIP's limit is 77)
    
    Returns:
        List of text chunks
    """
    # Split by spaces to get words
    words = text.split()
    
    # Create chunks of words that will likely fit within token limit
    # On average, each word is 1-2 tokens
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for word in words:
        if current_word_count >= max_tokens // 2:  # Conservative estimate: 2 words per token
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_word_count = 1
        else:
            current_chunk.append(word)
            current_word_count += 1
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def get_clip_text_embedding(text, model, device):
    """Generate CLIP embedding for text, handling long text by chunking and aggregating"""
    # Split text into chunks that fit CLIP's token limit
    chunks = split_text_into_chunks(text)
    
    if len(chunks) > 1:
        print(f"Text split into {len(chunks)} chunks due to length")
    
    # Process each chunk and collect embeddings
    chunk_embeddings = []
    for i, chunk in enumerate(chunks):
        print(f"Processing text chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
        try:
            with torch.no_grad():
                text_tokens = clip.tokenize([chunk]).to(device)
                features = model.encode_text(text_tokens)
                chunk_embeddings.append(features.cpu().numpy())
        except RuntimeError as e:
            print(f"Error encoding text chunk: {str(e)}")
            print(f"Skipping problematic chunk: {chunk[:50]}...")
            continue
    
    if not chunk_embeddings:
        print("Warning: Failed to generate any valid text embeddings")
        return None
    
    # Aggregate chunk embeddings using mean pooling
    aggregated_embedding = np.mean(chunk_embeddings, axis=0)
    print(f"Generated aggregated text embedding from {len(chunk_embeddings)} chunks")
    
    return aggregated_embedding

def aggregate_embeddings(embeddings):
    return np.mean(embeddings, axis=0, keepdims=True)

def find_videos_recursive(data_folder):
    """Find all mp4 files in the data folder and its subdirectories recursively."""
    video_paths = []
    
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                video_paths.append(video_path)
    
    return video_paths

def get_category_from_path(video_path, data_folder):
    """Extract category from video path."""
    # Get the relative path from data_folder
    rel_path = os.path.relpath(video_path, data_folder)
    # The first part of the path is the category
    parts = rel_path.split(os.sep)
    
    if len(parts) > 1:
        return parts[0]
    else:
        return "uncategorized"

def get_text_file_path(video_path):
    """Get the corresponding txt file path for a video."""
    base_path = os.path.splitext(video_path)[0]
    txt_path = base_path + '.txt'
    return txt_path

def read_text_file(txt_path):
    """Read text from a txt file, returning empty string if file doesn't exist."""
    if os.path.exists(txt_path):
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading txt file {txt_path}: {e}")
            return ""
    else:
        print(f"Warning: No corresponding txt file found at {txt_path}")
        return ""

def process_all_videos(data_folder="Data", num_frames=None, frame_interval=1, batch_size=32):
    video_paths = find_videos_recursive(data_folder)
    
    if not video_paths:
        print(f"No mp4 files found in {data_folder} folder or its subdirectories")
        return
    
    print(f"Found {len(video_paths)} videos to process:")
    for path in video_paths:
        category = get_category_from_path(path, data_folder)
        print(f" - [{category}] {os.path.basename(path)}")
    
    # Get the best available device
    device_type = print_gpu_info()
    print(f"Using device: {device_type}")
    device = torch.device(device_type)
    
    # Load CLIP model once for all videos
    print("Loading CLIP model...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Process each video
    results = []
    for video_path in video_paths:
        category = get_category_from_path(video_path, data_folder)
        print(f"\n==== Processing [{category}] {os.path.basename(video_path)} ====")
        
        # Extract frames
        frames = extract_frames(video_path, num_frames, frame_interval)
        
        if not frames:
            print(f"Warning: No frames were extracted from {video_path}")
            continue
            
        # Get video filename without path and extension
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Generate embeddings for all extracted frames
        frame_features = get_clip_embeddings(frames, model, preprocess, device, batch_size)
        
        # Aggregate embeddings using mean strategy
        print("Aggregating embeddings using mean strategy...")
        aggregated_features = aggregate_embeddings(frame_features)
        
        # Process corresponding text file if it exists
        txt_path = get_text_file_path(video_path)
        text_content = read_text_file(txt_path)
        text_features = None
        
        if text_content:
            text_features = get_clip_text_embedding(text_content, model, device)
            if text_features is not None:
                print(f"Generated text embedding (shape: {text_features.shape})")
        
        # Save embeddings
        timestamp = int(time.time())
        output_dir = os.path.join("embeddings", category)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save frame embeddings
        frames_output_path = os.path.join(output_dir, f"{video_name}_embeddings_{timestamp}.npy")
        np.save(frames_output_path, frame_features)
        print(f"Saved all {len(frame_features)} frame embeddings to {frames_output_path}")
        
        # Save aggregated embeddings
        aggregated_output_path = os.path.join(output_dir, f"{video_name}_aggregated_mean_{timestamp}.npy")
        np.save(aggregated_output_path, aggregated_features)
        print(f"Saved aggregated embeddings to {aggregated_output_path} (shape: {aggregated_features.shape})")
        
        # Save text embeddings if available
        text_output_path = None
        if text_features is not None:
            text_output_path = os.path.join(output_dir, f"{video_name}_text_{timestamp}.npy")
            np.save(text_output_path, text_features)
            print(f"Saved text embeddings to {text_output_path} (shape: {text_features.shape})")
        
        result = {
            "video_name": video_name,
            "category": category,
            "frame_embeddings": frame_features,
            "frame_embeddings_path": frames_output_path,
            "aggregated_embeddings": aggregated_features,
            "aggregated_embeddings_path": aggregated_output_path,
            "text_content": text_content if text_content else None,
            "text_embeddings": text_features,
            "text_embeddings_path": text_output_path
        }
        
        results.append(result)
    
    return results

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Generate CLIP embeddings for videos")
    parser.add_argument("--data", default="Data", help="Folder containing the videos and category subfolders")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames to extract. If not specified, extract all frames")
    parser.add_argument("--interval", type=int, default=8, help="Frame interval when extracting all frames (default: every 8th frame)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for processing through CLIP")
    
    args = parser.parse_args()
    
    # Print system info
    print(f"Python version: {sys.version}")
    
    # Process all videos with the specified parameters
    results = process_all_videos(
        args.data, 
        args.frames, 
        args.interval, 
        args.batch
    )
    
    # Print summary
    if results:
        print("\n==== Summary ====")
        print(f"Processed {len(results)} videos")
        for result in results:
            video_name = result["video_name"]
            category = result["category"]
            frame_embeddings = result["frame_embeddings"]
            aggregated_embeddings = result["aggregated_embeddings"]
            has_text = result["text_content"] is not None
            
            print(f" - [{category}] {video_name}: {frame_embeddings.shape[0]} frames, {frame_embeddings.shape[1]} dimensions")
            print(f"   Aggregated shape: {aggregated_embeddings.shape}")
            if has_text:
                print(f"   Text embedding: {result['text_embeddings'].shape}")
    
    print("\nDone!") 