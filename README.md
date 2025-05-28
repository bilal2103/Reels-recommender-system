# ReelsRS - Video Recommendation System

This project uses CLIP (Contrastive Language-Image Pre-training) to generate embeddings for videos and their associated text descriptions, enabling content-based recommendations and similarity searches for short video clips.

## Features

### Video Processing and Embedding Generation
- Extract frames from video files with customizable frame count and interval
- Generate CLIP embeddings for video frames
- Process video captions/descriptions and generate text embeddings
- Aggregation of frame embeddings with different strategies (mean, max)
- Support for multiple video categories
- GPU acceleration with CUDA and MPS support

### Similarity Search
- Find similar videos based on both visual and textual content
- Customizable weighting between visual and textual similarity
- Support for cross-category recommendations
- Ranking of most similar content

### Embedding Visualization
- 2D and 3D visualization of embeddings using t-SNE or PCA
- Visualization of different embedding types (frame, aggregated, text)
- Color-coded visualization by video category
- Support for interactive exploration of the embedding space

### API Server
- FastAPI-based RESTful API for video recommendations
- CORS support for cross-origin requests

## Setup and Installation

### Requirements

Install all required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Generate Embeddings

```bash
python clip_video_embeddings.py --data_folder Data --num_frames 10 --frame_interval 3
```

The script will:
1. Find all videos in the specified data folder (organized by category)
2. Extract frames from each video
3. Generate CLIP embeddings for frames and associated text descriptions
4. Save embeddings to the "embeddings" folder organized by category

### Find Similar Videos

```bash
python find_similar_videos.py Data/Category/video.mp4 --k 5 --video-weight 0.7 --text-weight 0.3
```

This command will:
1. Load embeddings for the specified video
2. Find the top 5 most similar videos across all categories
3. Display results with similarity scores

### Visualize Embeddings

```bash
python visualize_embeddings.py --embedding-type aggregated --dimensions 2 --method tsne
```

This will:
1. Load embeddings of the specified type (frame, aggregated, or text)
2. Reduce dimensions using t-SNE or PCA
3. Create a visualization with videos color-coded by category

### Start the API Server

```bash
python app.py
```

The FastAPI server will start, allowing access to the recommendation system via HTTP requests.

## API Endpoints

- `GET /` - Root endpoint (welcome message)
- Additional endpoints for recommendation functionality can be added as needed

## Customization

You can modify the scripts to:
- Process different video formats
- Change the embedding models or parameters
- Customize similarity metrics
- Adjust visualization settings 

## Conclusion

ReelsRS represents a cutting-edge approach to video recommendation, leveraging the power of CLIP's multimodal embeddings to understand both visual and textual content. By combining advanced frame processing, sophisticated embedding generation, and flexible similarity search capabilities, this system offers a robust solution for content-based video recommendations. Whether you're building a social media platform, content discovery system, or video analytics tool, ReelsRS provides the foundation for intelligent, content-aware video recommendations that go beyond traditional metadata-based approaches. The system's modular architecture, comprehensive API, and visualization tools make it both powerful and adaptable, ready to evolve with your specific use case and scale with your needs. 