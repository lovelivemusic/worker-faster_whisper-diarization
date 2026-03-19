# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **RunPod.md**: Documentation for RunPod serverless billing and idle timeout optimization
  - Explains cost discrepancy between app display and actual billing
  - Recommended idle timeout settings for different use cases
  - Cost optimization tips

## [1.2.0] - 2026-03-17

### Added
- `verbose` input parameter for detailed per-request logging
- Diarization pipeline caching at worker startup for faster processing
- Documentation for reducing hallucinations in README

### Fixed
- Audio download failure now returns clean error message instead of crashing

## [1.1.0] - 2026-03-17

### Added
- HF_TOKEN setup instructions in README
- Startup debugging logs for easier troubleshooting

### Fixed
- JSON serialization for numpy types (float64, int64) preventing 400 errors
- Diarization segment serialization with explicit float/int conversion
- Word timestamps serialization
- Added matplotlib dependency required by pyannote.audio

### Changed
- Default model changed from `base` to `large-v3`
- Pinned torch==2.4.0 and torchaudio==2.4.0 for pyannote.audio 3.3.1 compatibility
- Downgraded to CUDA 12.4.1 for stability

## [1.0.0] - 2026-03-15

### Added
- Speaker diarization using pyannote.audio 3.3.1
- `diarize` input parameter to control diarization (default: true)
- Support for audio URL and base64 audio input
- Word-level timestamps with `word_timestamps` parameter
- Multiple transcription formats: plain_text, formatted_text, srt, vtt
- Translation support with `translate` parameter
- VAD filtering with `enable_vad` parameter
- Configurable Whisper parameters: beam_size, temperature, language, etc.

### Changed
- Migrated from Whisper to faster-whisper for improved performance
- Docker base image updated to CUDA 12.4.1 with cuDNN

### Infrastructure
- RunPod serverless worker deployment
- Pre-loaded pyannote models in Docker image
- GPU-accelerated inference with CUDA support
