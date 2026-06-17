- #### the repo contains high-quality TTS (Text-to-Speech) training dataset of 60 minutes total duration —  30 minutes of Indian English and 30 minutes of any Indian language of your choice.

- ####  120 samples of ~30 seconds total ~60 minutes.

-----------------------------------------------------------------------------------

#### HF Dataset Link :
<p>
https://huggingface.co/datasets/rtxtd/tts-dataset-guj-eng
</p>

-------------------------------------------------------------------------------------
#### Contains : 

- Clean, single-speaker audio segments sourced from YouTube
- Accurate transcriptions
- Emotion/style tags for each segment (e.g., happy, sad, excited, angry, neutral, formal, whisper, etc.)
- Published as apublic HuggingFace dataset

#### tools used 

- Sarvam ASR APIs (saaras v3 model) for speaker diarization, asr, and generating transcripts 
- Used Sarvam 105b for emotion labelling of audio segments like sad, happy,excited,angry  etc. also the style conversationsl, storytelling etc. 

#### Environment variables required:
    SARVAM_API_KEY  
    HF_TOKEN        
    HF_REPO_ID     

| Feature                    | Details                                       |
| -------------------------- | --------------------------------------------- |
| **Total Duration**         | 60 minutes                                    |
| **Number of Samples**      | 120 audio segments                            |
| **Segment Length**         | 30 seconds                                    |
| **Languages**              | Indian English (30 min) + Gujarati (30 min)   |
| **Speaker Type**           | Clean, single-speaker recordings              |
| **Source**                 | YouTube audio segments                        |
| **Format**                 | TTS-ready audio + transcripts + metadata      |
