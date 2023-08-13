from potassium import Potassium, Request, Response
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import torchaudio

app = Potassium("my_app")

@app.init
def init():
    # Initialize the AudioGen model
    audio_model = AudioGen.get_pretrained('facebook/audiogen-medium')
    audio_model.set_generation_params(duration=5)  # generate 5 seconds of audio.

    context = {
        "audio_model": audio_model,
        "hello": "world"
    }

    return context

@app.handler()
def handler(context: dict, request: Request) -> Response:
    # Retrieve the audio model from the app context
    audio_model = context.get("audio_model")

    # Obtain descriptions from the request (ensure your client sends descriptions as a list)
    descriptions = request.json.get("descriptions", [])

    # Generate audio based on descriptions
    wav = audio_model.generate(descriptions)

    # Save generated audio files and collect filenames
    filenames = []
    for idx, one_wav in enumerate(wav):
        filename = f'{idx}.wav'
        audio_write(filename, one_wav.cpu(), audio_model.sample_rate, strategy="loudness", loudness_compressor=True)
        filenames.append(filename)
    
    # Return the filenames in the response (modify as per your requirements)
    return Response(
        json = {"audio_files": filenames}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()
