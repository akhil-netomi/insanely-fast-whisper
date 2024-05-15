from openai import OpenAI

def get_asr_from_openai(api_key: str, file_name: str):
    client = OpenAI(api_key=api_key)
    audio_file = open(file_name, "rb")
    transcript = client.audio.transcriptions.create(
    file=audio_file,
    model="whisper-1",
    response_format="verbose_json",
    timestamp_granularities=["segment"]
    )
    speech2text_output = []
    for o in transcript.segments: speech2text_output.append(dict(text=o["text"], timestamp=[o["start"], o["end"]]))
    return dict(chunks=speech2text_output, text=transcript.text)