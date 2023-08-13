from potassium import Potassium, Request, Response
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import torchaudio
import io
import base64

app = Potassium("my_app")

@app.init
def init():
    # Initialize the AudioGen model
    audio_model = AudioGen.get_pretrained('facebook/audiogen-medium')
    audio_model.set_generation_params(duration=5)  # generate 5 seconds of audio.

    context = {
        "audio_model": audio_model
    }

    return context

@app.handler()
def handler(context: dict, request: Request) -> Response:
    # Retrieve the audio model from the app context
    audio_model = context.get("audio_model")

    # Obtain descriptions from the request
    descriptions = request.json.get("descriptions", [])

    # Generate audio based on descriptions
    wav = audio_model.generate(descriptions)

    # Convert audio tensors to base64 encoded WAV data and associate with descriptions
    audio_data = []
    for desc, one_wav in zip(descriptions, wav):
        buffer = io.BytesIO()
        torchaudio.save(buffer, one_wav.cpu(), sample_rate=audio_model.sample_rate, format="wav")
        base64_audio = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        audio_data.append({
            "description": desc,
            "audio": base64_audio
        })

    # Return the combined data in the response
    return Response(
        json = {"audio_data": audio_data}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()
