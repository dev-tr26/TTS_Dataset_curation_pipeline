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


```
tts_dataset_pipeline/
│
├── configs/
│   └── pipeline.yaml          # all thresholds, API settings, weights
│
├── src/
│   ├── schemas.py             # all Pydantic data models
│   ├── ingestion/
│   │   └── ingest.py          # Stage 1: yt-dlp download + provenance
│   ├── preprocessing/
│   │   └── audio.py           # Stage 2+3: ffmpeg extraction + speech/music filtering
│   ├── diarization/
│   │   └── diarize.py         # Stage 4+5: Sarvam Batch API diarization + segment cutting
│   ├── transcription/
│   │   └── transcribe.py      # Stage 6+7: Sarvam ASR + 4-layer transcript validation
│   ├── validation/
│   │   └── validate.py        # re-export of TranscriptValidator
│   ├── emotion/
│   │   └── label.py           # Stage 9: acoustic + Sarvam LLM emotion labeling
│   ├── quality/
│   │   ├── assess.py          # Stage 8: SNR, clipping, silence, speech rate scoring
│   │   └── scorer.py          # Stage 11: weighted final quality score
│   |
├── scripts/
│   ├── run_pipeline.py        # main orchestrator: runs all stages end-to-end
│   ├── package_dataset.py     # run after review: packages + pushes to HF
│   └── publish_dataset.py     # standalone: label emotions + publish existing JSONs
│
├── tests/
│   └── test_pipeline.py       # unit tests for schemas, scoring, validation, audio
│
├── transcripts/               # final dataset output
│   ├── english/
│   │   ├── en_transcripts.json
│   │   └── *.wav
│   └── gujarati/
│       ├── gu_transcripts.json
│       └── *.wav
│
├── videos.csv                 # input: YouTube URLs + language
├── requirements.txt
└── README.md
```
