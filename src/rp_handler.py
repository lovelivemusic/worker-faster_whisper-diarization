"""
rp_handler.py for runpod worker
"""
print("=== HANDLER STARTING ===", flush=True)

import base64
import subprocess
import tempfile
from pathlib import Path

print("Basic imports done", flush=True)

print("Importing pyannote...", flush=True)
from pyannote.audio import Pipeline
print("Pyannote imported OK", flush=True)

from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate
import runpod
import predict
import torch
import numpy as np

print("All imports done", flush=True)

np.NAN = np.nan

print("Setting up model...", flush=True)
MODEL = predict.Predictor()
MODEL.setup()
print("Model setup complete", flush=True)

# Cache diarization pipeline at startup
print("Loading diarization pipeline...", flush=True)
DIARIZE_PIPELINE = Pipeline.from_pretrained('config.yaml')
DIARIZE_PIPELINE.to(torch.device('cuda'))
print("Diarization pipeline loaded", flush=True)


def base64_to_tempfile(base64_file: str) -> str:
    '''
    Convert base64 file to tempfile.

    Parameters:
    base64_file (str): Base64 file

    Returns:
    str: Path to tempfile
    '''
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))

    return temp_file.name


def _to_wav(fpath):
    path = Path(fpath)
    new_name = path.name.split('.')[0] + '.wav'
    new_path = path.parent / new_name
    subprocess.run([
        'ffmpeg',
        '-i', str(fpath),
        '-ar', '16000',
        '-ac', '1',
        '-c:a', 'pcm_s16le',
        str(new_path)
    ])
    return new_path


def diarize(fpath):
    if not str(fpath).lower().endswith('.wav'):
        fpath = _to_wav(fpath)

    resp = {'segments': []}
    dia = DIARIZE_PIPELINE(fpath)

    speakers = {}
    for turn, _, speaker in dia.itertracks(yield_label=True):
        if speaker not in speakers:
            speakers[speaker] = len(speakers)  # assign ordered index

        segdata = {'start': float(turn.start), 'end': float(turn.end), 'speaker': speakers[speaker]}
        resp['segments'].append(segdata)

    return resp


@rp_debugger.FunctionTimer
def run_whisper_job(job):
    '''
    Run inference on the model.

    Parameters:
    job (dict): Input job containing the model parameters

    Returns:
    dict: The result of the prediction
    '''
    job_input = job['input']

    with rp_debugger.LineTimer('validation_step'):
        input_validation = validate(job_input, INPUT_VALIDATIONS)

        if 'errors' in input_validation:
            return {"error": input_validation['errors']}
        job_input = input_validation['validated_input']

    verbose = job_input.get('verbose', False)

    if verbose:
        print(f"[VERBOSE] Job ID: {job['id']}", flush=True)
        print(f"[VERBOSE] Input params: model={job_input['model']}, language={job_input['language']}, "
              f"diarize={job_input['diarize']}, word_timestamps={job_input['word_timestamps']}", flush=True)
        print(f"[VERBOSE] Whisper params: beam_size={job_input['beam_size']}, temperature={job_input['temperature']}, "
              f"no_speech_threshold={job_input['no_speech_threshold']}, condition_on_previous_text={job_input['condition_on_previous_text']}", flush=True)

    if not job_input.get('audio', False) and not job_input.get('audio_base64', False):
        return {'error': 'Must provide either audio or audio_base64'}

    if job_input.get('audio', False) and job_input.get('audio_base64', False):
        return {'error': 'Must provide either audio or audio_base64, not both'}

    if job_input.get('audio', False):
        with rp_debugger.LineTimer('download_step'):
            audio_input = download_files_from_urls(job['id'], [job_input['audio']])[0]
            if audio_input is None:
                return {'error': f"Failed to download audio from: {job_input['audio']}"}
            if verbose:
                print(f"[VERBOSE] Downloaded audio to: {audio_input}", flush=True)

    if job_input.get('audio_base64', False):
        audio_input = base64_to_tempfile(job_input['audio_base64'])
        if verbose:
            print(f"[VERBOSE] Decoded base64 audio to: {audio_input}", flush=True)

    if verbose:
        print("[VERBOSE] Starting transcription...", flush=True)

    with rp_debugger.LineTimer('prediction_step'):
        resp = MODEL.predict(
            audio=audio_input,
            model_name=job_input["model"],
            transcription=job_input["transcription"],
            translation=job_input["translation"],
            translate=job_input["translate"],
            language=job_input["language"],
            temperature=job_input["temperature"],
            best_of=job_input["best_of"],
            beam_size=job_input["beam_size"],
            patience=job_input["patience"],
            length_penalty=job_input["length_penalty"],
            suppress_tokens=job_input.get("suppress_tokens", "-1"),
            initial_prompt=job_input["initial_prompt"],
            condition_on_previous_text=job_input["condition_on_previous_text"],
            temperature_increment_on_fallback=job_input["temperature_increment_on_fallback"],
            compression_ratio_threshold=job_input["compression_ratio_threshold"],
            logprob_threshold=job_input["logprob_threshold"],
            no_speech_threshold=job_input["no_speech_threshold"],
            enable_vad=job_input["enable_vad"],
            word_timestamps=job_input["word_timestamps"],
            repetition_penalty=job_input["repetition_penalty"],
            no_repeat_ngram_size=job_input["no_repeat_ngram_size"],
        )

    if verbose:
        print(f"[VERBOSE] Transcription complete. Segments: {len(resp.get('segments', []))}", flush=True)

    if job_input['diarize']:
        if verbose:
            print("[VERBOSE] Starting diarization...", flush=True)
        resp['diarization'] = diarize(audio_input)
        if verbose:
            print(f"[VERBOSE] Diarization complete. Speaker segments: {len(resp['diarization'].get('segments', []))}", flush=True)

    with rp_debugger.LineTimer('cleanup_step'):
        rp_cleanup.clean(['input_objects'])

    if verbose:
        print("[VERBOSE] Job complete.", flush=True)

    return resp


runpod.serverless.start({"handler": run_whisper_job})
