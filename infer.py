import os
import sys
import numpy as np
import tensorflow as tf
import torch
import clip
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import argparse
from PIL import Image
import shutil
from typing import List, Tuple

class TransNetV2:
    """
    TransNetV2 model for shot boundary detection.
    This class detects scene transitions in videos.
    """
    def __init__(self, model_dir=None):
        model_dir = "transnetv2-weights"
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "transnetv2-weights/")
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(f"[TransNetV2] ERROR: {model_dir} is not a directory.")
            else:
                print(f"[TransNetV2] Using weights from {model_dir}.")
        # Configure TensorFlow to limit GPU memory usage
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Limit TensorFlow to use only 30% of GPU memory
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    # Alternatively, set a memory limit:
                    # tf.config.experimental.set_virtual_device_configuration(
                    #     gpu,
                    #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=24576)]  # 24GB
                    # )
                print("[TransNetV2] TensorFlow GPU memory growth enabled")
            except RuntimeError as e:
                print(f"[TransNetV2] Error setting GPU memory growth: {e}")
                
        self._input_size = (27, 48, 3)
        try:
            self._model = tf.saved_model.load(model_dir)
        except OSError as exc:
            raise IOError(f"[TransNetV2] It seems that files in {model_dir} are corrupted or missing. "
                         f"Re-download them manually and retry. For more info, see: "
                         f"https://github.com/soCzech/TransNetV2/issues/1#issuecomment-647357796") from exc

    def predict_raw(self, frames: np.ndarray):
        assert len(frames.shape) == 5 and frames.shape[2:] == self._input_size, \
            "[TransNetV2] Input shape must be [batch, frames, height, width, 3]."
        frames = tf.cast(frames, tf.float32)

        logits, dict_ = self._model(frames)
        single_frame_pred = tf.sigmoid(logits)
        all_frames_pred = tf.sigmoid(dict_["many_hot"])

        return single_frame_pred, all_frames_pred

    def predict_frames(self, frames: np.ndarray):
        assert len(frames.shape) == 4 and frames.shape[1:] == self._input_size, \
            "[TransNetV2] Input shape must be [frames, height, width, 3]."

        def input_iterator():
            # return windows of size 100 where the first/last 25 frames are from the previous/next batch
            # the first and last window must be padded by copies of the first and last frame of the video
            no_padded_frames_start = 25
            no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)  # 25 - 74

            start_frame = np.expand_dims(frames[0], 0)
            end_frame = np.expand_dims(frames[-1], 0)
            padded_inputs = np.concatenate(
                [start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end, 0
            )

            ptr = 0
            while ptr + 100 <= len(padded_inputs):
                out = padded_inputs[ptr:ptr + 100]
                ptr += 50
                yield out[np.newaxis]

        predictions = []

        for inp in input_iterator():
            single_frame_pred, all_frames_pred = self.predict_raw(inp)
            predictions.append((single_frame_pred.numpy()[0, 25:75, 0],
                              all_frames_pred.numpy()[0, 25:75, 0]))

            print("\r[TransNetV2] Processing video frames {}/{}".format(
                min(len(predictions) * 50, len(frames)), len(frames)
            ), end="")
        print("")

        single_frame_pred = np.concatenate([single_ for single_, all_ in predictions])
        all_frames_pred = np.concatenate([all_ for single_, all_ in predictions])

        return single_frame_pred[:len(frames)], all_frames_pred[:len(frames)]  # remove extra padded frames

    def predict_video(self, video_fn: str):
        try:
            import ffmpeg
        except ModuleNotFoundError:
            raise ModuleNotFoundError("For `predict_video` function `ffmpeg` needs to be installed in order to extract "
                                     "individual frames from video file. Install `ffmpeg` command line tool and then "
                                     "install python wrapper by `pip install ffmpeg-python`.")

        print("[TransNetV2] Extracting frames from {}".format(video_fn))
        video_stream, err = ffmpeg.input(video_fn).output(
            "pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27"
        ).run(capture_stdout=True, capture_stderr=True)

        video = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])
        return (video, *self.predict_frames(video))

    @staticmethod
    def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5):
        predictions = (predictions > threshold).astype(np.uint8)

        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])

        # just fix if all predictions are 1
        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)

        return np.array(scenes, dtype=np.int32)


class VideoKeyframeExtractor:
    """
    Main class for extracting keyframes from videos based on LMSKE approach.
    """
    def __init__(self, transnet_weights=None, output_dir="keyframes", sample_rate=1, max_frames_per_shot=100):
        """
        Initialize the keyframe extractor.
        
        Args:
            transnet_weights: Path to TransNetV2 weights
            output_dir: Directory to save keyframes
            sample_rate: Sample every Nth frame to reduce computation
            max_frames_per_shot: Maximum number of frames to process per shot
        """
        self.transnet = TransNetV2(transnet_weights)
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.max_frames_per_shot = max_frames_per_shot
        
        # Initialize CLIP model for feature extraction
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[CLIP] Using device: {self.device}")
        self.model, self.preprocess = clip.load('ViT-B/32', self.device)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_video_frames(self, video_path: str) -> Tuple[List[np.ndarray], List[int]]:
        """
        Extract frames from a video at the specified sample rate.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (frames, frame_indices)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        frames = []
        frame_indices = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % self.sample_rate == 0:
                # Convert from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_indices.append(frame_idx)
                
            frame_idx += 1
            
        cap.release()
        return frames, frame_indices
    
    def extract_clip_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract CLIP features for a list of frames.
        
        Args:
            frames: List of frames in RGB format
            
        Returns:
            Array of feature vectors
        """
        features = []
        batch_size = 32  # Process in batches to avoid memory issues
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            
            # Convert frames to PIL Images and preprocess for CLIP
            batch_inputs = torch.stack([
                self.preprocess(Image.fromarray(frame)) 
                for frame in batch_frames
            ]).to(self.device)
            
            # Extract features
            with torch.no_grad():
                batch_features = self.model.encode_image(batch_inputs)
                batch_features /= batch_features.norm(dim=-1, keepdim=True)  # Normalize
            
            features.append(batch_features.cpu().numpy())
            print(f"\r[CLIP] Processing frames {i+len(batch_frames)}/{len(frames)}", end="")
            
        print("")
        return np.vstack(features)
    
    def adaptive_clustering(self, features: np.ndarray) -> List[int]:
        """
        Perform adaptive clustering to determine keyframes as described in the paper.
        
        Args:
            features: Frame feature vectors
            
        Returns:
            Indices of selected keyframes
        """
        n_samples = features.shape[0]
        if n_samples <= 1:
            return [0] if n_samples == 1 else []
            
        # If very few frames, return all of them
        if n_samples <= 3:
            return list(range(n_samples))
        
        # Calculate maximum number of clusters based on sqrt(n)
        k_max = min(int(np.sqrt(n_samples)), 10)  # Cap at 10 clusters maximum
        
        # Try different numbers of clusters and find the best using silhouette score
        best_score = -1
        best_k = 2  # Minimum 2 clusters
        best_labels = None
        
        for k in range(2, k_max + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            
            if len(np.unique(labels)) > 1:  # Ensure we have at least 2 clusters
                score = silhouette_score(features, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels
                    best_centers = kmeans.cluster_centers_
        
        if best_labels is None:
            # Fallback to 2 clusters if silhouette score fails
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            best_labels = kmeans.fit_predict(features)
            best_centers = kmeans.cluster_centers_
        
        # Select frames closest to cluster centers
        keyframe_indices = []
        for i in range(len(best_centers)):
            cluster_frames = np.where(best_labels == i)[0]
            if len(cluster_frames) > 0:
                # Find the frame closest to the cluster center
                dists = np.linalg.norm(features[cluster_frames] - best_centers[i], axis=1)
                closest_frame = cluster_frames[np.argmin(dists)]
                keyframe_indices.append(closest_frame)
        
        # Remove similar keyframes (eliminating redundancy)
        # Calculate pairwise distances between keyframe features
        keyframe_features = features[keyframe_indices]
        n_keyframes = len(keyframe_indices)
        
        if n_keyframes <= 1:
            return keyframe_indices
            
        # Calculate similarity matrix
        similarity_matrix = np.zeros((n_keyframes, n_keyframes))
        for i in range(n_keyframes):
            for j in range(i+1, n_keyframes):
                # Cosine similarity
                similarity = np.dot(keyframe_features[i], keyframe_features[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        # Iteratively remove redundant frames
        to_keep = list(range(n_keyframes))
        threshold = 0.8  # Similarity threshold
        
        i = 0
        while i < len(to_keep):
            j = i + 1
            while j < len(to_keep):
                if similarity_matrix[to_keep[i], to_keep[j]] > threshold:
                    # Remove the jth keyframe
                    to_keep.pop(j)
                else:
                    j += 1
            i += 1
        
        return [keyframe_indices[i] for i in to_keep]
    
    def extract_keyframes(self, video_path: str) -> None:
        """
        Main function to extract keyframes from a video.
        
        Args:
            video_path: Path to the video file
        """
        # Create directory for this video's keyframes
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(self.output_dir, video_name)
        
        if os.path.exists(video_output_dir):
            shutil.rmtree(video_output_dir)
        os.makedirs(video_output_dir)
        
        print(f"[KeyframeExtractor] Processing {video_path}")
        
        # Step 1: Run TransNetV2 to get shot boundaries
        _, single_frame_predictions, _ = self.transnet.predict_video(video_path)
        scenes = self.transnet.predictions_to_scenes(single_frame_predictions)
        
        print(f"[KeyframeExtractor] Detected {len(scenes)} shots in the video")
        
        # Step 2: Process each shot to extract keyframes
        all_keyframes = []
        
        # Extract all frames at the sample rate
        print("[KeyframeExtractor] Extracting frames from video")
        all_frames, all_frame_indices = self.extract_video_frames(video_path)
        
        # Process each shot
        for shot_idx, (start_idx, end_idx) in enumerate(scenes):
            print(f"[KeyframeExtractor] Processing shot {shot_idx+1}/{len(scenes)} (frames {start_idx}-{end_idx})")
            
            # Find frames within this shot based on frame indices
            shot_frame_indices = [
                i for i, frame_idx in enumerate(all_frame_indices)
                if start_idx <= frame_idx <= end_idx
            ]
            
            # If too many frames in shot, sample them
            if len(shot_frame_indices) > self.max_frames_per_shot:
                step = len(shot_frame_indices) // self.max_frames_per_shot
                shot_frame_indices = shot_frame_indices[::step]
            
            if not shot_frame_indices:
                print(f"[KeyframeExtractor] Warning: No frames found for shot {shot_idx+1}")
                continue
                
            # Get the actual frames
            shot_frames = [all_frames[i] for i in shot_frame_indices]
            
            # Extract CLIP features for this shot's frames
            print(f"[KeyframeExtractor] Extracting features for {len(shot_frames)} frames in shot {shot_idx+1}")
            shot_features = self.extract_clip_features(shot_frames)
            
            # Apply adaptive clustering to get keyframes
            keyframe_indices = self.adaptive_clustering(shot_features)
            
            # Map back to original frame indices
            shot_keyframe_indices = [shot_frame_indices[i] for i in keyframe_indices]
            
            # Save the keyframes
            for i, frame_idx in enumerate(shot_keyframe_indices):
                original_frame_idx = all_frame_indices[frame_idx]
                keyframe_path = os.path.join(
                    video_output_dir, 
                    f"shot_{shot_idx+1:03d}_keyframe_{i+1:02d}_frame_{original_frame_idx:05d}.jpg"
                )
                
                # Save the image
                Image.fromarray(all_frames[frame_idx]).save(keyframe_path)
                all_keyframes.append((keyframe_path, shot_idx, i, original_frame_idx))
        
        print(f"[KeyframeExtractor] Extracted {len(all_keyframes)} keyframes from {len(scenes)} shots")
        print(f"[KeyframeExtractor] Keyframes saved to {video_output_dir}")
        
        # Create a summary file
        with open(os.path.join(video_output_dir, "keyframes_summary.txt"), "w") as f:
            f.write(f"Video: {video_path}\n")
            f.write(f"Total shots: {len(scenes)}\n")
            f.write(f"Total keyframes: {len(all_keyframes)}\n\n")
            
            for keyframe_path, shot_idx, keyframe_idx, original_frame_idx in all_keyframes:
                filename = os.path.basename(keyframe_path)
                f.write(f"{filename}: Shot {shot_idx+1}, Keyframe {keyframe_idx+1}, Original Frame {original_frame_idx}\n")


def download_transnetv2_weights():
    """
    Download TransNetV2 weights if they don't exist.
    """
    weights_dir = os.path.join(os.path.dirname(__file__), "transnetv2-weights")
    if os.path.exists(weights_dir):
        print(f"[TransNetV2] Weights already exist at {weights_dir}")
        return weights_dir
        
    print("[TransNetV2] Downloading weights...")
    import urllib.request
    import zipfile
    
    # Create the directory
    os.makedirs(weights_dir, exist_ok=True)
    
    # Download the weights
    url = "https://github.com/soCzech/TransNetV2/releases/download/v1.0/transnetv2-weights.zip"
    zip_path = os.path.join(os.path.dirname(__file__), "transnetv2-weights.zip")
    
    try:
        urllib.request.urlretrieve(url, zip_path)
        
        # Extract the weights
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(__file__))
            
        # Remove the zip file
        os.remove(zip_path)
        print(f"[TransNetV2] Weights downloaded and extracted to {weights_dir}")
        return weights_dir
    except Exception as e:
        print(f"[TransNetV2] Error downloading weights: {e}")
        print("[TransNetV2] Please download the weights manually from https://github.com/soCzech/TransNetV2/releases")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Extract keyframes from videos using LMSKE approach")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("--output", type=str, default="keyframes", help="Output directory for keyframes")
    parser.add_argument("--sample-rate", type=int, defaul, help="Sample every N frames to reduce computation")
    parser.add_argument("--max-frames", type=int, default=50, help="Maximum number of frames to process per shot")
    args = parser.parse_args()
    
    # Download TransNetV2 weights if necessary
    # weights_dir = download_transnetv2_weights()
    
    # Initialize and run the keyframe extractor
    extractor = VideoKeyframeExtractor(
        transnet_weights="/home/liex/Desktop/Work/keyframe_extraction/TransNetV2/inference/transnetv2-weights",
        output_dir=args.output,
        sample_rate=args.sample_rate,
        max_frames_per_shot=args.max_frames
    )
    
    extractor.extract_keyframes(args.video_path)


if __name__ == "__main__":
    main()
# python infer.py /root/NL/L01_V001.mp4 --output /root/NL/TransNetV2/inference/extracted_frames
