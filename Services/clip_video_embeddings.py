import os
import sys
# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import cv2
from PIL import Image
import time
import clip
import glob
import argparse
# Add imports for MongoDB integration
from Models.Reel import Reel
from Services.MongoService import MongoService

def extract_frames(video_path, num_frames=None, frame_interval=8):
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
    # For hashtag-heavy content, split by hashtags first
    if text.count('#') > 3:  # If there are multiple hashtags
        chunks = []
        # First chunk: everything before hashtags (if exists)
        main_text = text.split('#')[0].strip()
        if main_text:
            chunks.append(main_text)
        
        # Then process hashtags, grouping them in smaller bundles
        hashtag_parts = ['#' + part for part in text.split('#')[1:] if part.strip()]
        current_chunk = ""
        
        for hashtag in hashtag_parts:
            # If adding this hashtag would make chunk too long, start a new chunk
            if len(current_chunk + " " + hashtag) > 30:  # Conservative estimate
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = hashtag
            else:
                current_chunk = (current_chunk + " " + hashtag).strip()
        
        # Add the last hashtag chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    # Regular text splitting for non-hashtag-heavy content
    words = text.split()
    chunks = []
    current_chunk = []
    
    # More conservative splitting: ~1.5 tokens per word on average
    words_per_chunk = max_tokens // 2  
    
    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i:i + words_per_chunk])
        chunks.append(chunk)
    
    return chunks

def get_clip_text_embedding(text, model, device):
    """Generate CLIP embedding for text, handling long text by chunking and aggregating"""
    if not text:
        print("Warning: Empty text provided, skipping text embedding")
        return None
        
    # Split text into chunks that fit CLIP's token limit
    chunks = split_text_into_chunks(text)
    
    if len(chunks) > 1:
        print(f"Text split into {len(chunks)} chunks due to length")
    
    # Process each chunk and collect embeddings
    chunk_embeddings = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():  # Skip empty chunks
            continue
            
        print(f"Processing text chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
        
        # Try with incrementally shorter versions if needed
        success = False
        attempt_chunk = chunk
        
        for attempt in range(3):  # Try up to 3 times with shorter text
            try:
                with torch.no_grad():
                    text_tokens = clip.tokenize([attempt_chunk]).to(device)
                    features = model.encode_text(text_tokens)
                    chunk_embeddings.append(features.cpu().numpy())
                    success = True
                    break
            except RuntimeError as e:
                # If it's too long, try with a shorter version (70% of original)
                if "context length" in str(e) and attempt < 2:
                    words = attempt_chunk.split()
                    new_length = int(len(words) * 0.7)
                    attempt_chunk = " ".join(words[:new_length])
                    print(f"  Retrying with shorter text ({new_length} words)")
                else:
                    print(f"  Error encoding text chunk: {str(e)}")
                    print(f"  Skipping problematic chunk: {chunk[:50]}...")
                    break
    
    if not chunk_embeddings:
        print("Warning: Failed to generate any valid text embeddings")
        # Return a default embedding filled with zeros as fallback
        default_embedding = np.zeros((1, 512))  # CLIP ViT-B/32 uses 512 dimensions
        return default_embedding
    
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

def process_all_videos(model, preprocess,data_folder="Data", num_frames=None, frame_interval=1, batch_size=32):
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
    
    # Initialize MongoDB service
    mongo_service = MongoService()
    
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
        
        # Save embeddings to MongoDB
        print("Saving embeddings to MongoDB...")
        # Convert the numpy arrays to the correct format for MongoDB
        # For text features, ensure it's a 2D array before conversion
        if text_features is not None:
            # Ensure text_features is 2D
            if len(text_features.shape) == 1:
                text_features_list = text_features.reshape(1, -1).tolist()
            else:
                text_features_list = text_features.tolist()
        else:
            text_features_list = None

        # For aggregated features, ensure it's 2D
        if len(aggregated_features.shape) == 1:
            aggregated_features_list = aggregated_features.reshape(1, -1).tolist()
        else:
            aggregated_features_list = aggregated_features.tolist()

        reel = Reel(
            path=video_path,
            category=category,
            videoEmbeddings=frame_features.tolist() if frame_features is not None else None,
            aggregatedEmbeddings=aggregated_features_list,
            textualEmbeddings=text_features_list
        )
        
        reel_id = mongo_service.AddReel(reel)
        print(f"Saved reel to MongoDB with ID: {reel_id}")
        
        result = {
            "id": reel_id,
            "video_name": video_name,
            "category": category,
            "frame_embeddings": frame_features,
            "aggregated_embeddings": aggregated_features,
            "text_content": text_content if text_content else None,
            "text_embeddings": text_features
        }
        
        results.append(result)
    
    return results

def calculate_similarity_matrix(results):
    """Calculate cosine similarity between all pairs of video embeddings.
    
    Args:
        results: List of dictionaries containing the embedding results
        
    Returns:
        similarity_matrix: 2D numpy array of similarity scores
        video_names: List of video names corresponding to the matrix indices
    """
    # Extract aggregated embeddings and video names
    embeddings = []
    video_names = []
    categories = []
    
    for result in results:
        # Reshape to 1D if needed
        embedding = result["aggregated_embeddings"].flatten()
        embeddings.append(embedding)
        video_names.append(result["video_name"])
        categories.append(result["category"])
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings)
    
    # Normalize the embeddings to calculate cosine similarity
    normalized_embeddings = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    
    # Calculate cosine similarity
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    
    return similarity_matrix, video_names, categories

def print_similarity_matrix(similarity_matrix, video_names, categories, n_closest=3):
    n_videos = len(video_names)
    
    print("\n==== Video Similarity Matrix ====")
    print(f"Comparing {n_videos} videos\n")
    
    # Print top N most similar videos for each video
    for i in range(n_videos):
        # Get similarities for this video, excluding self-similarity
        similarities = similarity_matrix[i].copy()
        similarities[i] = -1  # Exclude self
        
        # Get indices of top N most similar videos
        top_indices = np.argsort(similarities)[::-1][:n_closest]
        
        print(f"[{categories[i]}] {video_names[i]}:")
        for idx in top_indices:
            similarity = similarity_matrix[i, idx]
            print(f"  - {similarity:.4f} similar to [{categories[idx]}] {video_names[idx]}")
        print()

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Generate CLIP embeddings for videos")
    parser.add_argument("--data", default="Data", help="Folder containing the videos and category subfolders")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames to extract. If not specified, extract all frames")
    parser.add_argument("--interval", type=int, default=8, help="Frame interval when extracting all frames (default: every 8th frame)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for processing through CLIP")
    parser.add_argument("--similar", type=int, default=3, help="Number of most similar videos to display for each video")
    parser.add_argument("--video", help="Path to a specific video file to process (instead of processing all videos)")
    
    args = parser.parse_args()
    
    # Print system info
    print(f"Python version: {sys.version}")
    
    # Get the best available device
    device_type = print_gpu_info()
    print(f"Using device: {device_type}")
    device = torch.device(device_type)
    
    # Load CLIP model once for all videos
    print("Loading CLIP model...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Initialize MongoDB service
    mongo_service = MongoService()
    
    results = []
    
    # Process a single video if specified, otherwise process all videos
    if args.video:
        video_path = args.video
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} does not exist")
            sys.exit(1)
            
        # Determine category from path
        data_folder = args.data
        # Default to uncategorized if can't determine
        category = "uncategorized"
        
        try:
            # Try to extract category from path
            rel_path = os.path.relpath(video_path, data_folder)
            parts = rel_path.split(os.sep)
            if len(parts) > 1:
                category = parts[0]
        except:
            # If there's an error determining category, keep default
            pass
            
        print(f"\n==== Processing [{category}] {os.path.basename(video_path)} ====")
        
        # Extract frames
        frames = extract_frames(video_path, args.frames, args.interval)
        
        if not frames:
            print(f"Warning: No frames were extracted from {video_path}")
            sys.exit(1)
            
        # Get video filename without path and extension
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Generate embeddings for all extracted frames
        frame_features = get_clip_embeddings(frames, model, preprocess, device, args.batch)
        
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
        
        # Save embeddings to MongoDB
        print("Saving embeddings to MongoDB...")
        # Convert the numpy arrays to the correct format for MongoDB
        # For text features, ensure it's a 2D array before conversion
        if text_features is not None:
            # Ensure text_features is 2D
            if len(text_features.shape) == 1:
                text_features_list = text_features.reshape(1, -1).tolist()
            else:
                text_features_list = text_features.tolist()
        else:
            text_features_list = None

        # For aggregated features, ensure it's 2D
        if len(aggregated_features.shape) == 1:
            aggregated_features_list = aggregated_features.reshape(1, -1).tolist()
        else:
            aggregated_features_list = aggregated_features.tolist()

        reel = Reel(
            path=video_path,
            category=category,
            videoEmbeddings=frame_features.tolist() if frame_features is not None else None,
            aggregatedEmbeddings=aggregated_features_list,
            textualEmbeddings=text_features_list
        )
        
        reel_id = mongo_service.AddReel(reel)
        print(f"Saved reel to MongoDB with ID: {reel_id}")
        
        result = {
            "id": reel_id,
            "video_name": video_name,
            "category": category,
            "frame_embeddings": frame_features,
            "aggregated_embeddings": aggregated_features,
            "text_content": text_content if text_content else None,
            "text_embeddings": text_features
        }
        
        results.append(result)
        
        print(f"\n==== Single Video Processing Complete ====")
        print(f"Video: {video_name}")
        print(f"Category: {category}")
        print(f"Frames processed: {len(frame_features)}")
        print(f"Embedding dimensions: {frame_features.shape[1]}")
        print(f"Text processed: {'Yes' if text_content else 'No'}")
        print(f"MongoDB ID: {reel_id}")
        
    else:
        # Process all videos
        results = process_all_videos(
            model,
            preprocess,
            args.data, 
            args.frames, 
            args.interval, 
            args.batch
        )
    
    # Print summary
    if results and len(results) > 1:
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
            print(f"   MongoDB ID: {result['id']}")
            if has_text:
                print(f"   Text embedding: {result['text_embeddings'].shape}")
        
        # Calculate and print similarity matrix
        similarity_matrix, video_names, categories = calculate_similarity_matrix(results)
        print_similarity_matrix(similarity_matrix, video_names, categories, n_closest=args.similar)
    
    print("\nDone!") 