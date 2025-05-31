## Setup

```
pip install -r requirements.txt
pip install -U bitsandbytes 
```
**Install git lfs**
```
sudo apt-get install git-lfs
git lfs install
```
## Download weight
```
cd Keyframe_Extraction
git clone https://huggingface.co/xinyu1205/recognize-anything-plus-model
```
## Run
`python infer.py <path_to_video.mp4> --output <path_to_output_folder>
`
